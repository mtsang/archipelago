import sys
import numpy as np
import pickle
import requests
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm


def prep_imagenet_coco_conversion(
    coco,
    i1k_labels_url="https://s3.amazonaws.com/outcome-blog/imagenet/labels.json",
    data_dir="/meladyfs/newyork/datasets/mscoco",
    data_type="val2017",
    coco_to_i1k_path="processed_data/image_data/coco_to_i1k_map.pickle",
):
    # get imagenet labels
    i1k_labels = {
        int(key): value for (key, value) in requests.get(i1k_labels_url).json().items()
    }
    i1k_labels_rev = {v: k for k, v in i1k_labels.items()}

    # maps coco labels to imagenet labels
    with open(coco_to_i1k_path, "rb") as handle:
        coco_to_i1k_map = pickle.load(handle)

    # get ms coco data
    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    cat_nms = [cat["name"] for cat in cats]
    # print('COCO categories: \n{}\n'.format(' '.join(cat_nms)))

    supercat_nms = set([cat["supercategory"] for cat in cats])
    # print('COCO supercategories: \n{}'.format(' '.join(supercat_nms)))

    # maps a category id to its name and the name of its supercategory
    cat_map = {cat["id"]: (cat["name"], cat["supercategory"]) for cat in cats}

    # get category ids that intersect with imagenet labels
    valid_cats = list(coco_to_i1k_map.keys())
    valid_cat_ids = coco.getCatIds(catNms=[k[0] for k in valid_cats])

    # maps imagenet label indices to coco categories
    i1k_idx_to_cat = {}
    for nm in valid_cats:
        for i1k_label in coco_to_i1k_map[nm]:
            i1k_idx = i1k_labels_rev[i1k_label[0]]
            if i1k_idx not in i1k_idx_to_cat:
                i1k_idx_to_cat[i1k_idx] = set()
            i1k_idx_to_cat[i1k_idx].add(nm)

    return i1k_idx_to_cat, valid_cat_ids, cat_map


# pytorch transformation functions for preprocessing images
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ]
)
preprocess_mask = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def transform_img(image, preprocess, device):
    image_tensor = preprocess(Image.fromarray(image))
    denom = image_tensor.abs().max().item()
    image = image_tensor.cpu().numpy().transpose(1, 2, 0) / denom
    image_tensor = image_tensor.unsqueeze_(0).to(device) / denom
    return image, image_tensor


def match_segments_and_mask(segments, mask_orig, ratio_threshold=0.5):
    inter = []
    for seg in np.unique(segments):
        seg_mask = 1 * (segments == seg)
        # if the original mask overlaps with > 50% of a superpixel segment, count that segment as
        # part of the mask
        ratio = (seg_mask * mask_orig[:, :, 0]).sum() / seg_mask.sum()
        if ratio > ratio_threshold:
            inter.append(seg)
    return inter


def generate_perturbation_dataset_bert(
    data_inst,
    model,
    class_idx,
    device,
    num_samples=6000,
    batch_size=100,
    seed=None,
    model_id=None,
    **kwargs
):
    sys.path.append("../../../baselines/mahe_madex/madex/")
    from utils.general_utils import set_seed, proprocess_data
    from sampling_and_inference import generate_binary_perturbations

    if seed is not None:
        set_seed(seed)

    target_ids = data_inst["target"]
    baseline_ids = data_inst["baseline"]
    samples_binary = generate_binary_perturbations(len(target_ids), num_samples, True)
    n_batches = int(np.ceil(num_samples / batch_size))
    samples_labels = []
    for i in tqdm(range(n_batches)):
        samples_binary_batch = samples_binary[i * batch_size : (i + 1) * batch_size]
        perturbed_text = []
        for sample_binary in samples_binary_batch:
            vec = target_ids.copy()
            vec[sample_binary == 0] = baseline_ids[sample_binary == 0]

            perturbed_text.append(vec)
        preds = (
            model(torch.LongTensor(np.stack(perturbed_text)).to(device))[0]
            .data.cpu()
            .numpy()
        )
        samples_labels.append(preds)
    samples_labels = np.concatenate(samples_labels)
    Xs, Ys = proprocess_data(samples_binary, samples_labels[:, class_idx], **kwargs)

    return Xs, Ys


def convert_spans_to_interactions(spans):
    inters = []
    for span in spans:
        inter = tuple(range(span[0], span[1] + 1))
        inters.append(inter)
    return inters
