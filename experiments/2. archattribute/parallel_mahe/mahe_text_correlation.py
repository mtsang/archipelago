import sys, os
import pickle
import torch
from transformers import *
import torch.multiprocessing as multiprocessing
from itertools import repeat
from tqdm import tqdm

sys.path.append("..")
from experiment_utils import *

sys.path.append("../../../src")
from application_utils.text_utils import *

sys.path.append("../../../baselines/mahe_madex/madex/")
sys.path.append("../../../baselines/mahe_madex/mahe/")
from deps.interaction_explainer import learn_hierarchical_gam


gt_file = "../processed_data/text_data/subtree_allphrase_nosentencelabel.pickle"
save_path = "../analysis/results/phrase_corr_mahe.pickle"

# gt_file = '../processed_data/text_data/subtree_single_token.pickle'
# save_path = "../analysis/results/word_corr_mahe.pickle"

model_path = "../../../downloads/pretrained_bert"
num_processes = 10

device = torch.device("cuda:0")
mlp_device = torch.device("cuda:0")


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

    with open(gt_file, "rb") as handle:
        phrase_gt_splits = pickle.load(handle)

    phrase_gt = phrase_gt_splits["test"]
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)

    class_idx = 1

    if os.path.exists(save_path):
        with open(save_path, "rb") as handle:
            p_dict = pickle.load(handle)
        ref = p_dict["ref"]
        est_methods = p_dict["est"]
    else:
        ref = {}
        est_methods = {"mahe": {}}

    for s_idx, phrase_dict in enumerate(tqdm(phrase_gt)):

        if all((s_idx in est_methods[m]) for m in ["mahe"]) and s_idx in ref:
            print("skip", s_idx)
            continue

        sentence = phrase_dict["sentence"]
        tokens = phrase_dict["tokens"]
        subtrees = phrase_dict["subtrees"]
        att_len = len(tokens)

        span_to_label = {}
        for subtree in subtrees:
            span_to_label[subtree["span"]] = subtree["label"]

        spans = list(span_to_label.keys())

        baseline_token = "_"
        text_ids, baseline_ids = get_input_baseline_ids(
            sentence, baseline_token, tokenizer
        )

        data_inst = {"target": text_ids, "baseline": baseline_ids}

        inters = convert_spans_to_interactions(spans)

        Xs, Ys = generate_perturbation_dataset_bert(
            data_inst,
            model,
            class_idx,
            device,
            batch_size=15,
            #                 num_samples = 100,
            #                 test_size=10,
            #                 valid_size=10,
            seed=s_idx,
        )

        for k in Xs:
            Xs[k] = Xs[k][:, 1:-1]

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
            label = span_to_label[spans[i]]
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

        est_methods["mahe"][s_idx] = est_vec

        ref[s_idx] = ref_vec

        with open(save_path, "wb") as handle:
            pickle.dump(
                {"est": est_methods, "ref": ref},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


if __name__ == "__main__":
    run()
