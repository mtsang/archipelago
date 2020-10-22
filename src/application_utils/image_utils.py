from torchvision import transforms
import requests
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
from application_utils.common_utils import get_efficient_mask_indices
from PIL import Image
from skimage.segmentation import find_boundaries
from skimage.segmentation import quickshift, watershed, slic
from skimage.morphology import dilation, square
from skimage.util import img_as_float


class ImageXformer:  # (data_xformer):
    def __init__(self, input_img, baseline_img, segments):
        self.input = input_img
        self.baseline = baseline_img
        self.segments = segments
        self.num_features = len(np.unique(segments))

    def simple_xform(self, inst):
        mask_indices = np.argwhere(inst == True).flatten()
        mask = np.isin(self.segments, mask_indices)
        image = self.baseline.copy()
        image[mask] = self.input[mask]
        return image

    def efficient_xform(self, inst):
        mask_indices, base, change = get_efficient_mask_indices(
            inst, self.baseline, self.input
        )
        mask = np.isin(self.segments, mask_indices)
        base[mask] = change[mask]
        return base

    def __call__(self, inst):
        return self.efficient_xform(inst)


def get_image_and_labels(
    image_path,
    device,
    labels_url="https://s3.amazonaws.com/outcome-blog/imagenet/labels.json",
):
    """
    Loads image instance and labels

    Args:
        image_path: path to image instance
        labels_url: url to json labels

    Returns:
        image, labels
    """
    # image pre-processing needed for ResNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_tensor = preprocess(image)
    image = (
        image_tensor.cpu().numpy().transpose(1, 2, 0) / image_tensor.abs().max().item()
    )
    labels = {
        int(key): value for (key, value) in requests.get(labels_url).json().items()
    }
    return image, labels


def plot_explanations(img_arrays, figsize=0.4, spacing=0.15, savepath=""):
    w_spacing = (2 / 3) * spacing
    left = 0
    ax_arays = []
    fig = plt.figure()
    for img_array in img_arrays:
        num_imgs = len(img_array)
        right = left + figsize * (num_imgs) + (num_imgs - 1) * 0.4 * w_spacing
        ax_arays.append(
            fig.subplots(
                1, num_imgs, gridspec_kw=dict(left=left, right=right, wspace=w_spacing)
            )
        )
        left = right + spacing

    for i, ax_array in enumerate(ax_arays):
        if hasattr(ax_array, "flat"):
            for j, ax in enumerate(ax_array.flat):
                img, title = img_arrays[i][j]
                ax.imshow(img / 2 + 0.5)
                ax.set_title(title, fontsize=55 * figsize)
                ax.axis("off")
        else:
            img, title = img_arrays[i][0]

            ax_array.imshow(img / 2 + 0.5)
            ax_array.set_title(title, fontsize=55 * figsize)
            ax_array.axis("off")

    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


def get_set_img(image, segments, set_indices_instance, set_atts_instance, max_att):
    """
    Show archipelago sets and attributions
    """
    if not isinstance(set_atts_instance, list):
        set_atts_instance = [set_atts_instance] * len(set_indices_instance)
    temp = (np.ones(image.shape, image.dtype) - 0.5) * 1
    for i, n in enumerate(set_indices_instance):
        color_intensity = (
            0.6 * np.abs(set_atts_instance[i]) / (max_att) if max_att != 0 else 0
        )
        c = 0 if set_atts_instance[i] < 0 else 1
        temp[segments == n] = image[segments == n].copy() * (1 - color_intensity)
        temp[segments == n, c] += color_intensity
    return temp


def show_image_explanation(
    inter_effects,
    image,
    segments,
    figsize=0.4,
    spacing=0.15,
    main_effects=None,
    savepath="",
):
    """
    Format image visualizations for plotting
    """

    inter_sets, inter_atts = list(zip(*inter_effects))
    if main_effects is not None:
        main_id_list = []
        main_att_list = []
        for main_id, main_att in main_effects:
            main_id_list.append(main_id)
            main_att_list.append(main_att)

    max_att_main = np.amax(np.abs(main_att_list))
    max_att_inter = np.amax(np.abs(inter_atts))

    img_arrays = []
    img_arrays.append([(image, "Original image")])

    ## main effects
    if main_effects is not None:
        img_arrays.append(
            [
                (
                    get_set_img(
                        image, segments, main_id_list, main_att_list, max_att_main
                    ),
                    "Main effects",
                )
            ]
        )

    inter_img_arrays = []
    for i, inter_set in enumerate(inter_sets):
        inter_img_arrays.append(
            (
                get_set_img(image, segments, inter_set, inter_atts[i], max_att_inter),
                "Interaction $\mathcal{I}_" + str(i + 1) + "$",
            )
        )
    img_arrays.append(inter_img_arrays)

    plot_explanations(img_arrays, figsize, spacing, savepath)


def custom_mark_boundaries(
    image,
    label_img,
    color=(1, 1, 0),
    outline_color=None,
    mode="outer",
    background_label=0,
    outline_thickness=2,
):
    """Return image with boundaries between labeled regions highlighted.
    Parameters
    ----------
    image : (M, N[, 3]) array
        Grayscale or RGB image.
    label_img : (M, N) array of int
        Label array where regions are marked by different integer values.
    color : length-3 sequence, optional
        RGB color of boundaries in the output image.
    outline_color : length-3 sequence, optional
        RGB color surrounding boundaries in the output image. If None, no
        outline is drawn.
    mode : string in {'thick', 'inner', 'outer', 'subpixel'}, optional
        The mode for finding boundaries.
    background_label : int, optional
        Which label to consider background (this is only useful for
        modes ``inner`` and ``outer``).
    Returns
    -------
    marked : (M, N, 3) array of float
        An image in which the boundaries between labels are
        superimposed on the original image.
    See Also
    --------
    find_boundaries
    """
    marked = img_as_float(image, force_copy=True)
    if marked.ndim == 2:
        marked = gray2rgb(marked)
    if mode == "subpixel":
        # Here, we want to interpose an extra line of pixels between
        # each original line - except for the last axis which holds
        # the RGB information. ``ndi.zoom`` then performs the (cubic)
        # interpolation, filling in the values of the interposed pixels
        marked = ndi.zoom(
            marked, [2 - 1 / s for s in marked.shape[:-1]] + [1], mode="reflect"
        )
    boundaries = find_boundaries(label_img, mode=mode, background=background_label)
    if outline_color is not None:
        outlines = dilation(boundaries, square(outline_thickness))
        marked[outlines] = outline_color
    return marked


def overlay_explanation(explanation, image, segments, outline_thickness=2):
    """
    Show archipelago sets and attributions
    """
    max_att = np.max(np.abs(list(explanation.values())))

    temp = (np.ones(image.shape, image.dtype) - 0.5) * 1

    sorted_atts = sorted(explanation.items(), key=lambda item: -item[1])
    max_cnt = 5
    temp = image.copy()
    #     rgb = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (191, 255, 0), (64, 255, 0), (0, 255, 64), (0, 255, 191), (0, 191, 255), (0, 64, 255), (64, 0, 255)]
    #     rgb = [(0, 255, 0), (0, 255, 191), (0, 128, 255), (4, 0, 255), (255, 0, 255)]
    rgb = [(0, 255, 0), (0, 255, 191), (0, 191, 255), (128, 0, 255), (255, 0, 255)]

    sets_ranked, _ = zip(*sorted_atts)

    sets_ranked = [s for s, a in sorted_atts if len(s) > 1 and a > 0]
    for i in reversed(range(min(len(sets_ranked), max_cnt))):
        bound = np.isin(segments, sets_ranked[i]) * 1
        temp = custom_mark_boundaries(
            temp,
            bound,
            outline_color=np.array(rgb[i]) / 255,
            mode="inner",
            outline_thickness=outline_thickness,
        )

    num_pos_interactions = len(sets_ranked)

    return temp, num_pos_interactions
