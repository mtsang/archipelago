from torchvision import transforms
import requests
from PIL import Image
from skimage.segmentation import mark_boundaries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"


# image pre-processing needed for ResNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ]
)


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
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_tensor = preprocess(image)
    image = (
        image_tensor.cpu().numpy().transpose(1, 2, 0) / image_tensor.abs().max().item()
    )
    image_tensor = (
        image_tensor.unsqueeze_(0).to(device) / image_tensor.abs().max().item()
    )
    labels = {
        int(key): value for (key, value) in requests.get(labels_url).json().items()
    }
    return image, image_tensor, labels


def show_segmented_image(image, segments):
    plt.imshow(mark_boundaries(image / 2 + 0.5, segments))


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


def show_explanations(
    inter_sets, image, segments, figsize=0.4, spacing=0.15, lime_atts=None, savepath=""
):
    def get_interaction_img(inter):
        temp = (np.ones(image.shape, image.dtype) - 0.5) * 1
        for n in inter:
            temp[segments == n] = image[segments == n].copy()
        return temp

    img_arrays = []
    img_arrays.append([(image, "Original image")])

    ## main effects
    if lime_atts is not None:
        temp = (np.ones(image.shape, image.dtype) - 0.5) * 1
        for n, _ in lime_atts[:5]:
            temp[segments == n] = image[segments == n].copy()
        img_arrays.append([(temp, "Main effects")])

    inter_img_arrays = []
    for i, inter_set in enumerate(inter_sets):
        inter_img_arrays.append(
            (
                get_interaction_img(inter_set),
                "Interaction $\mathcal{I}_" + str(i + 1) + "$",
            )
        )
    img_arrays.append(inter_img_arrays)

    plot_explanations(img_arrays, figsize, spacing, savepath)
