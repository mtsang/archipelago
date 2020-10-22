from torchtext import datasets, data
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import sklearn
from time import time

from pipeline_mod import pipeline

sys.path.append("../madex")

from neural_interaction_detection import *
from sampling_and_inference import *
from utils.general_utils import *

from deps.interaction_explainer import learn_hierarchical_gam
from deps.lime_scores import get_lime_mse
import pickle

import torch.multiprocessing as multiprocessing
from itertools import repeat
import h5py as h5


from utils.dna_utils import *
from utils.text_utils import *
from utils.image_utils import *
from utils.graph_utils import *

from skimage.segmentation import quickshift
from torchvision import models

import argparse

import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--target_count", type=int, default=40)
parser.add_argument("--par_batch_size", type=int, default=10)
parser.add_argument("--num_trials", type=int, default=5)
parser.add_argument("--out_file", type=str, default="image_results.pickle")
parser.add_argument("--exp", type=str, help="experiment", default="image")
parser.add_argument(
    "--save_folder", type=str, help="path to save folder", default="experiments"
)
parser.add_argument("--valid_size", type=int, default=500)
parser.add_argument("--test_size", type=int, default=500)
parser.add_argument("--num_samples", type=int, default=6000)
parser.add_argument("--weight_samples", type=int, default=0)
parser.add_argument("--add_linear", type=int, default=0)
parser.add_argument("--model", type=str, default="default")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--mlp_gpu", type=int, default=1)


args = parser.parse_args()

target_count = args.target_count
par_batch_size = args.par_batch_size
num_trials = args.num_trials
out_file = args.out_file
experiment = args.exp
save_folder = args.save_folder
valid_size = args.valid_size
test_size = args.test_size
num_samples = args.num_samples
model_selector = args.model
weight_samples = args.weight_samples == 1
add_linear = args.add_linear == 1
gpu = args.gpu
mlp_gpu = args.mlp_gpu

if mlp_gpu < 0:
    mlp_device = torch.device("cpu")
else:
    mlp_device = torch.device("cuda:" + str(mlp_gpu))


def par_experiment(progress_id, index, preprocess_dict, num_trials):
    trials = []
    for trial in range(num_trials):
        print("progress_id", progress_id, "index", index, "trial", trial)
        preprocess_trial = preprocess_dict[index][trial]
        Xd = preprocess_trial["Xd"]
        Yd = preprocess_trial["Yd"]
        interactions, mlp_loss = detect_interactions(
            Xd,
            Yd,
            weight_samples=weight_samples,
            arch=[256, 128, 64],
            nepochs=100,
            add_linear=add_linear,
            early_stopping=True,
            patience=5,
            seed=trial,
            device=mlp_device,
            verbose=False,
        )
        lime_mse = get_lime_mse(Xd, Yd, weight_samples=weight_samples)
        (
            interaction_contributions,
            univariate_contributions,
            prediction_scores,
        ) = learn_hierarchical_gam(
            Xd,
            Yd,
            interactions,
            mlp_device,
            weight_samples=weight_samples,
            hierarchy_stepsize=4,
            num_steps=100,
            hierarchical_patience=2,
            nepochs=100,
            verbose=False,
            early_stopping=True,
            seed=trial,
        )
        trial_results = {
            "inter_contribs": interaction_contributions,
            "uni_contribs": univariate_contributions,
            "pred_scores": prediction_scores,
            "lime_loss": lime_mse,
            "nid_loss": mlp_loss,
        }
        trials.append(trial_results)

    return index, trials


def get_model_and_data(experiment, device, model_selector):

    vectorizer = None
    misc = None

    if experiment == "dna":

        def load_dna_data(h5file):
            f = h5.File(h5file, "r")
            g = f["data"]
            data = g["s_x"].value
            seqs = g["sequence"].value
            targets = g["c0_y"].value
            return (data, seqs, targets)

        model = load_dna_model("../utils/pretrained/dna_cnn.pt").to(device)
        _, test_data, test_labels = load_dna_data("../utils/data/Myc_test.h5")
        vectorizer = encode_dna_onehot

    elif experiment == "text":
        if model_selector == "bert":
            model = pipeline("sentiment-analysis", device=device)
            model.device = device
        else:
            model_folder = "../../1. madex/utils/pretrained"
            sys.path.insert(0, model_folder)
            model = get_text_model(model_folder + "/sentiment_lstm.pt").to(device)
        vectorizer = get_vectorizer()
        train_iter, dev_iter, test_iter = get_sst(device)

        test_data = []
        for batch_idx, batch in enumerate(test_iter):
            text, label = batch.text.data[:, 0], batch.label
            sentence = " ".join([vectorizer.vocab.itos[w] for w in text])
            test_data.append(sentence)

    elif experiment == "image":
        model = models.resnet152(pretrained=True).to(device).eval()
        #         base_path = "/meladyfs/newyork/datasets/imagenet14/test"
        base_path = "/home/mtsang/test"
        test_data = sorted(
            [base_path + "/" + f for f in os.listdir(base_path) if f.endswith(".JPEG")]
        )

    elif experiment == "graph":
        model_folder = "../../1. madex/utils/pretrained"
        model, n_nodes, n_hops, test_idxs = get_graph_model(model_folder)
        model = model.to(device)
        data_folder = "../../1. madex/utils/data/cora"
        node_feats, adj_mat, labels = load_cora(data_folder, device)
        test_data = test_idxs
        misc = (n_nodes, n_hops, node_feats, adj_mat)

    else:
        raise ValueError("Invalid experiment")

    return model, test_data, vectorizer, misc


def run():

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = save_folder + "/" + out_file

    multiprocessing.set_start_method("spawn", force=True)

    device = torch.device("cuda:" + str(gpu))

    model, test_data, vectorizer, misc = get_model_and_data(
        experiment, device, model_selector
    )

    set_seed(42)
    indexes = np.random.choice(len(test_data), target_count, replace=False)

    if os.path.exists(save_path):
        print("extending saved results in", out_file)
        with open(save_path, "rb") as handle:
            results = pickle.load(handle)
        indexes = np.array([i for i in indexes if i not in results])
    else:
        results = {}

    num_par_batches = int(np.ceil(len(indexes) / par_batch_size))

    for b in range(num_par_batches):
        index_batch = indexes[b * par_batch_size : (b + 1) * par_batch_size]
        progress_ids = list(
            range(b * par_batch_size, b * par_batch_size + len(index_batch))
        )
        preprocess_dict = {}
        for index in index_batch:
            data_instance = test_data[index]
            if experiment == "image":
                image, image_tensor, _ = get_image_and_labels(data_instance, device)
                predictions = model(image_tensor)
                class_idx = predictions.data.cpu().numpy()[0].argsort()[::-1][0]
                segments = quickshift(image, kernel_size=3, max_dist=300, ratio=0.2)

            preprocess_trials = []
            for trial in range(num_trials):

                if experiment == "dna":
                    data_inst = {"orig": data_instance, "vectorizer": vectorizer}
                    Xd, Yd = generate_perturbation_dataset_dna(
                        data_inst,
                        model,
                        device,
                        num_samples=num_samples,
                        batch_size=6000,
                        seed=trial,
                        valid_size=valid_size,
                        test_size=test_size,
                    )
                elif experiment == "text":
                    cls_idx = 1 if model_selector == "bert" else 0

                    data_inst = {"orig": data_instance, "vectorizer": vectorizer}
                    Xd, Yd = generate_perturbation_dataset_text(
                        data_inst,
                        model,
                        cls_idx,
                        device,
                        num_samples=num_samples,
                        model_id=model_selector,
                        batch_size=500,
                        seed=trial,
                        valid_size=valid_size,
                        test_size=test_size,
                    )
                elif experiment == "image":
                    data_inst = {"orig": image, "segments": segments}
                    Xd, Yd = generate_perturbation_dataset_image(
                        data_inst,
                        model,
                        class_idx,
                        device,
                        num_samples=num_samples,
                        batch_size=10,
                        seed=trial,
                        valid_size=valid_size,
                        test_size=test_size,
                    )
                elif experiment == "graph":
                    target_idx = data_instance
                    test_idxs = test_data
                    _, n_hops, node_feats, adj_mat = misc
                    data_inst = {
                        "nodes": node_feats,
                        "edges": adj_mat,
                        "test_idxs": test_idxs,
                    }
                    Xd, Yd = generate_perturbation_dataset_graph(
                        data_inst,
                        model,
                        target_idx,
                        n_hops,
                        device,
                        num_samples=num_samples,
                        batch_size=500,
                        seed=trial,
                        valid_size=valid_size,
                        test_size=test_size,
                    )
                else:
                    raise ValueError("Invalid experiment")

                preprocess_trials.append({"Xd": Xd, "Yd": Yd})
            preprocess_dict[index] = preprocess_trials

        with multiprocessing.Pool(processes=par_batch_size) as pool:
            results_batch = pool.starmap(
                par_experiment,
                zip(
                    progress_ids,
                    index_batch,
                    repeat(preprocess_dict),
                    repeat(num_trials),
                ),
            )

        for result_batch in results_batch:
            index, trials = result_batch
            results[index] = trials

        with open(save_path, "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    run()
