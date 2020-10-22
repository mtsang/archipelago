import numpy as np
import matplotlib.pyplot as plt
import csv
from utils.parser import get_span_to_node_mapping, parse_tree
import pickle
import argparse
import os


def len_span(span):
    return 1 if type(span) is int else span[1] - span[0] + 1


def compact_layer(layers):
    # collapse unimportant hierarchies
    items = sorted(list(layers.items()), key=lambda x: x[0])
    layers = [_[1] for _ in items]
    idx = 0
    compact_layers = {}
    skip_flg = False
    for i, layer in enumerate(layers):
        if (
            i != len(layers) - 1
            and len(layers[i]) == 1
            and len(layers[i + 1]) == 1
            and not skip_flg
        ):
            entry1, entry2 = layers[i][0], layers[i + 1][0]
            score1, score2 = entry1[3], entry2[3]
            start1, stop1 = entry1[2] - len(entry1[1]) + 1, entry1[2]
            start2, stop2 = entry2[2] - len(entry2[1]) + 1, entry2[2]
            if (stop2 - start2) - (stop1 - start1) == 1 and not score1 * score2 < 0:
                continue
        compact_layers[idx] = layer
        idx += 1
        skip_flg = False
    return compact_layers


def plot_score_array(layers, score_array, sent_words):
    max_abs = abs(score_array).max()
    width = max(10, score_array.shape[1])
    height = max(5, score_array.shape[0])
    fig, ax = plt.subplots(figsize=(width, height))

    vmin, vmax = -max_abs * 1.2, max_abs * 1.2
    im = ax.imshow(score_array, cmap="coolwarm", aspect=0.5, vmin=vmin, vmax=vmax)
    # y_ticks = [''] * score_array.shape[0]
    # y_ticks[0] = 'Atomics words'
    # y_ticks[-1] = 'Full sentence'
    ax.set_yticks([])
    ax.set_xticks([])
    # ax.set_yticks(np.arange(len(y_ticks)))
    # ax.set_yticklabels(y_ticks)
    rgba = im.cmap(im.norm(im.get_array()))
    cnt = 0
    if layers is not None:
        for idx, i in enumerate(sorted(layers.keys())):
            for entry in layers[i]:
                start, stop = entry[2] - len(entry[1]) + 1, entry[2]
                for j in range(start, stop + 1):
                    color = (0.0, 0.0, 0.0)
                    ax.text(
                        j,
                        cnt,
                        sent_words[j],
                        ha="center",
                        va="center",
                        fontsize=11 if len(sent_words[j]) < 10 else 8,
                        color=color,
                    )
            cnt += 1
    else:
        for i in range(score_array.shape[0]):
            for j in range(score_array.shape[1]):
                if score_array[i, j] != 0:
                    if sent_words[j].startswith("SUBJ") or sent_words[j].startswith(
                        "OBJ"
                    ):
                        sent_words[j] = (
                            sent_words[j]
                            .replace("SUBJ-", "S-")
                            .replace("OBJ-", "O-")
                            .upper()
                        )
                    fontsize = 12
                    if len(sent_words[j]) >= 8:
                        fontsize = 8
                    if len(sent_words[j]) >= 12:
                        fontsize = 6
                    ax.text(
                        j, i, sent_words[j], ha="center", va="center", fontsize=fontsize
                    )
    return im


def draw_tree_from_line(s, tree_s):
    stack = []
    layers = {}
    phrase_scores = s.strip().split("\t")
    word_offset = 0
    sent_words = []

    span2node, node2span = get_span_to_node_mapping(parse_tree(tree_s))
    spans = list(span2node.keys())

    for idx, span in enumerate(spans):
        items = phrase_scores[idx]
        words = span2node[span].leaves()
        score = float(items.split()[-1])
        if len(words) == 1:
            layer = 0
            entry = (layer, words, word_offset, score)
            word_offset += 1
            sent_words.append(words[0])
            stack.append(entry)
        else:
            len_sum, max_layer = 0, -1
            while len_sum < len(words):
                popped = stack[-1]
                max_layer = max(popped[0], max_layer)
                len_sum += len(popped[1])
                stack.pop(-1)
            layer = max_layer + 1
            entry = (layer, words, word_offset - 1, score)
            stack.append(entry)

        if layer not in layers:
            layers[layer] = []
        layers[layer].append(entry)

    layers = compact_layer(layers)

    score_array = []

    for layer in sorted(layers.keys()):
        arr = np.zeros(len(sent_words))
        for entry in layers[layer]:
            start, stop = entry[2] - len(entry[1]) + 1, entry[2]  # closed interval
            arr[start : stop + 1] = entry[3]
        score_array.append(arr)

    score_array = np.stack(score_array)

    im = plot_score_array(layers, score_array, sent_words)
    return im, score_array


def visualize_tree(result_path, model_name, method_name, tree_path):
    f = open(result_path)
    f2 = open(tree_path)
    os.makedirs("figs/fig{}_{}".format(model_name, method_name), exist_ok=True)
    for i, (line, tree_str) in enumerate(zip(f.readlines(), f2.readlines())):
        im, score_array = draw_tree_from_line(line, tree_str)
        plt.savefig(
            "figs/fig{}_{}/{}".format(model_name, method_name, i), bbox_inches="tight"
        )
        plt.close()


def visualize_tab(tab_file_dir, model_name, method_name):
    f = open(tab_file_dir, "rb")
    data = pickle.load(f)
    os.makedirs("figs/{}_{}".format(model_name, method_name), exist_ok=True)
    for i, entry in enumerate(data):
        sent_words = entry["text"].split()
        score_array = entry["tab"]
        label_name = entry["label"]
        if score_array.ndim == 1:
            score_array = score_array.reshape(1, -1)
        if score_array.shape[1] <= 400:
            im = plot_score_array(None, score_array, sent_words)
            plt.title(label_name)
            plt.savefig(
                "figs/fig{}_{}/{}".format(model_name, method_name, i),
                bbox_inches="tight",
            )
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="pkl or txt output by the explanation algorithm")
    parser.add_argument("--model", help="model name for saving images")
    parser.add_argument("--method", help="method name for saving images")
    parser.add_argument(
        "--use_gt_trees",
        help="whether use ground truth parsing trees (sst)"
        "if true, --file should be .txt",
        action="store_true",
    )
    parser.add_argument("--gt_tree_path", default=".data/sst/trees/%s.txt" % "test")
    args = parser.parse_args()
    if not args.use_gt_trees:
        visualize_tab(args.file, args.model, args.method)
    else:
        visualize_tree(args.file, args.model, args.method, args.gt_tree_path)
