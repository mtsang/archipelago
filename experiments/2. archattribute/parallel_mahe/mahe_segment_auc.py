import os, sys
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from torchvision import models
from tqdm import tqdm

sys.path.append("..")
from experiment_utils import *

sys.path.append("../../../src")
from application_utils.image_utils import *

sys.path.append("../../../baselines/mahe_madex/madex/")
sys.path.append("../../../baselines/mahe_madex/mahe/")
from utils.general_utils import set_seed, proprocess_data
from sampling_and_inference import generate_perturbation_dataset_image
from deps.interaction_explainer import learn_hierarchical_gam
import torch.multiprocessing as multiprocessing
from itertools import repeat

import pickle
import warnings
from time import time

warnings.filterwarnings("ignore")
device = torch.device("cuda:0")
mlp_device = torch.device("cuda:0")

from pycocotools.coco import COCO
import skimage.io as io
from skimage.color import gray2rgb
import pylab


num_processes = 6
softmax = False

img_exp_path = "../"
save_path = img_exp_path + "analysis/results/segment_auc_mahe.pickle"
coco_to_i1k_path = img_exp_path + "processed_data/image_data/coco_to_i1k_map.pickle"


data_dir = "/meladyfs/newyork/datasets/mscoco"
data_type = "val2017"


def par_experiment(index, Xd, Yd, interaction, mlp_device):
    # modify mlp_device to distribute device load, e.g. mlp_device = index % 2

    if interaction == None:
        interactions = []
    else:
        interactions = [(interaction, 0)]

    (
        interaction_contributions,
        univariate_contributions,
        prediction_scores,
    ) = learn_hierarchical_gam(
        Xd,
        Yd,
        interactions,
        mlp_device,
        weight_samples=True,
        hierarchy_stepsize=4,
        num_steps=100,
        hierarchical_patience=2,
        nepochs=100,
        verbose=False,
        early_stopping=True,
        stopping=False,
        seed=index,
    )

    trial_results = {
        "inter_contribs": interaction_contributions,
        "uni_contribs": univariate_contributions,
        "pred_scores": prediction_scores,
    }
    return index, trial_results


def run():
    multiprocessing.set_start_method("spawn", force=True)

    model = models.resnet152(pretrained=True).to(device).eval()

    annFile = "{}/annotations/instances_{}.json".format(data_dir, data_type)
    coco = COCO(annFile)
    i1k_idx_to_cat, valid_cat_ids, cat_map = prep_imagenet_coco_conversion(
        coco, data_dir=data_dir, data_type=data_type, coco_to_i1k_path=coco_to_i1k_path
    )

    show_plots = False
    max_imgs_per_category = 500

    if os.path.exists(save_path):
        with open(save_path, "rb") as handle:
            results = pickle.load(handle)
    else:
        results = {}
    seenImgs = set()

    t0 = time()
    img_count = 0

    for cat_id in tqdm(valid_cat_ids):

        imgIds = coco.getImgIds(catIds=[cat_id])
        coco_label = cat_map[cat_id]

        for _, imgId in enumerate(imgIds):

            if imgId in seenImgs:
                continue
            seenImgs.add(imgId)
            img = coco.loadImgs(imgId)[0]
            annIds = coco.getAnnIds(
                imgIds=img["id"], catIds=valid_cat_ids, iscrowd=None
            )

            if imgId in results and all(
                len(results[imgId]["est"][m]) == len(annIds) for m in ["mahe", "mahe2"]
            ):  #### here i set the number of anns
                continue

            anns = coco.loadAnns(annIds)
            I = io.imread("%s/images/%s/%s" % (data_dir, data_type, img["file_name"]))

            if len(I.shape) == 2:
                I = gray2rgb(I)
            if show_plots:
                plt.imshow(I)
                plt.axis("off")

            image = Image.fromarray(I)
            image, image_tensor = transform_img(I, preprocess, device)
            top_model_class_idxs = (
                model(image_tensor.to(device)).data.cpu().numpy()[0].argsort()[::-1]
            )

            for i in top_model_class_idxs:
                if i in i1k_idx_to_cat:
                    model_target_idx = i
                    break

            segments = quickshift(image, kernel_size=3, max_dist=300, ratio=0.2)
            data_inst = {"orig": image, "segments": segments}
            results[imgId] = {"ref": [], "est": {}}

            inters = []
            ann_labels = []
            for ann in anns:
                assert ann["category_id"] in valid_cat_ids

                mask = coco.annToMask(ann)

                ## maintain black mask... (dont normalize)
                mask_resize = np.tile(np.expand_dims(mask, 2), 3).astype(np.uint8)

                mask_orig, mask_tensor = transform_img(
                    mask_resize, preprocess_mask, device
                )
                if math.isnan(mask_tensor.sum().item()):
                    continue

                inter = match_segments_and_mask(segments, mask_orig)

                if len(inter) == 0:
                    continue

                if cat_map[ann["category_id"]] not in i1k_idx_to_cat[model_target_idx]:
                    ann_labels.append(0)
                else:
                    ann_labels.append(1)
                inters.append(tuple(inter))

            if not inters:
                continue

            inters2 = []
            i1_to_i2 = {}
            for i, inter in enumerate(inters):
                if len(inter) == 1:
                    if None not in inters2:
                        i1_to_i2[i] = len(inters2)
                        non_idx = len(inters2)
                        inters2.append(None)
                    else:
                        i1_to_i2[i] = non_idx

                else:
                    i1_to_i2[i] = len(inters2)
                    inters2.append(inter)

            Xs, Ys = generate_perturbation_dataset_image(
                data_inst,
                model,
                model_target_idx,
                device,
                num_samples=6000,
                batch_size=10,
                seed=img_count,
                valid_size=500,
                test_size=500,
            )

            with multiprocessing.Pool(processes=num_processes) as pool:
                results_batch = pool.starmap(
                    par_experiment,
                    zip(
                        list(range(len(inters2))),
                        repeat(Xs),
                        repeat(Ys),
                        inters2,
                        repeat(mlp_device),
                    ),
                )

            results_dict = dict(results_batch)

            est_vec = []
            ref_vec = []

            for i, inter in enumerate(inters):
                label = ann_labels[i]
                ref_vec.append(label)
                rd = results_dict[i1_to_i2[i]]

                if len(inter) >= 2:
                    ires = rd["inter_contribs"]
                    est = ires[0][0][1]

                    for j in inter:
                        ures = rd["uni_contribs"]
                        est += ures[1][j]
                else:
                    ures = rd["uni_contribs"]
                    est = ures[1][inter[0]]

                est_vec.append(est)

            results[imgId]["ref"] = ref_vec
            results[imgId]["est"]["mahe"] = est_vec

            #             if img_count %  == 0:
            with open(save_path, "wb") as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            img_count += 1

    with open(save_path, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time()

    print("elapsed time", t1 - t0)


if __name__ == "__main__":
    run()
