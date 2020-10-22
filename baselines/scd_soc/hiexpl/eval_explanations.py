from utils.args import get_args
import torch
import numpy as np
from utils.reader import load_vocab
from bert.tokenization import BertTokenizer
from utils.parser import get_span_to_node_mapping, parse_tree
import csv, pickle
from collections import defaultdict
from utils.args import get_best_snapshot
from nns.linear_model import BOWRegression, BOWRegressionMulti
import argparse

args = get_args()
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", do_lower_case=True, cache_dir="bert/cache"
)


def unigram_linear_pearson(filename):
    f = open(filename)
    model = torch.load(args.bow_snapshot, map_location="cpu")
    vocab = load_vocab(VOCAB)
    out, truth = [], []
    coeff_dict = {}
    scores_dict = defaultdict(list)
    valid, total = 0, 0
    for lidx, line in enumerate(f.readlines()):
        if lidx < MINLINE:
            continue
        if lidx == MAXLINE:
            break
        l = line.lower().strip().split("\t")
        for entry in l:
            items = entry.strip().split(" ")
            if len(items) > 2:
                continue
            score = float(items[1])
            word = items[0]
            if word in vocab.stoi:
                coeff = -model.get_coefficient(vocab.stoi[word])
                out.append(score)
                truth.append(coeff)
                scores_dict[word].append(score)
                coeff_dict[word] = coeff
                valid += 1
            total += 1
    p = np.corrcoef(out, truth)
    print("word_corr", p[1, 0])

    out_2, truth_2 = [], []
    for k in scores_dict:
        out_2.extend([np.mean(scores_dict[k])])
        truth_2.extend([coeff_dict[k]])
    p2 = np.corrcoef(out_2, truth_2)
    print("average word_corr", p2[1, 0])
    return p[1, 0]


def unigram_linear_pearson_multiclass_agg(filename):
    f = open(filename, "rb")
    data = pickle.load(f)

    model = torch.load(args.bow_snapshot, map_location="cpu")
    vocab = load_vocab(VOCAB)
    out, truth = [], []

    valid, total = 0, 0
    for lidx, entry in enumerate(data):
        if lidx < MINLINE:
            continue
        if lidx == MAXLINE:
            break

        sent_words = entry["text"].split()
        score_array = entry["tab"]
        label_name = entry["label"]
        if score_array.ndim == 1:
            score_array = score_array.reshape(1, -1)
        for word, score in zip(sent_words, score_array[0].tolist()):
            if word in vocab.stoi:
                coeff = model.get_margin_coefficient(label_name, vocab.stoi[word])
                out.append(score)
                truth.append(coeff)
                valid += 1
            total += 1
    p = np.corrcoef(out, truth)
    print("word_corr", p[1, 0])
    return p[1, 0]


def unigram_linear_pearson_multiclass(filename):
    f = open(filename)
    model = torch.load(args.bow_snapshot, map_location="cpu")
    vocab = load_vocab(VOCAB)
    out, truth = [], []
    scores_dict, coeff_dict = defaultdict(list), defaultdict(float)

    valid, total = 0, 0
    for lidx, line in enumerate(f.readlines()):
        if lidx < MINLINE:
            continue
        if lidx == MAXLINE:
            break
        l = line.strip().split("\t")
        class_name = l[0]
        for entry in l[1:]:
            items = entry.strip().split(" ")
            if len(items) > 2:
                continue
            score = float(items[1])
            word = items[0]
            if word in vocab.stoi:
                coeff = model.get_label_coefficient(class_name, vocab.stoi[word])
                out.append(score)
                truth.append(coeff)
                scores_dict[word].append(score)
                coeff_dict[word] = coeff
                valid += 1
            total += 1
    p = np.corrcoef(out, truth)
    print("word_corr", p[1, 0])


def unigram_linear_pearson_bert_tree(filename, dataset="dev"):
    f = open(filename)
    f2 = open(".data/sst/trees/%s.txt" % dataset)
    model = torch.load(args.bow_snapshot, map_location="cpu")
    vocab = load_vocab(VOCAB)
    out, truth = [], []
    valid, total = 0, 0
    scores_dict = defaultdict(list)
    coeff_dict = defaultdict(int)
    coeff_not, coeff_not_cnt = 0, 0
    for lidx, (line, gt) in enumerate(zip(f.readlines(), f2.readlines()[OFFSET:])):
        if lidx < MINLINE:
            continue
        if lidx == MAXLINE:
            break
        entries = line.lower().strip().split("\t")
        span2node, node2span = get_span_to_node_mapping(parse_tree(gt))
        spans = list(span2node.keys())
        for idx, span in enumerate(spans):
            if type(span) is int:
                word = span2node[span].leaves()[0].lower()
                score = float(entries[idx].split()[-1])
                if word in vocab.stoi:
                    coeff = -model.get_coefficient(vocab.stoi[word])
                    out.append(score)
                    truth.append(coeff)
                    valid += 1
                    scores_dict[word].append(score)
                    coeff_dict[word] = coeff
                total += 1
    p = np.corrcoef(out, truth)
    print("word_corr", p[1, 0])

    out_2, truth_2 = [], []
    for k in scores_dict:
        out_2.extend([np.mean(scores_dict[k])])
        truth_2.extend([coeff_dict[k]])
    p2 = np.corrcoef(out_2, truth_2)
    print("avg word_corr", p2[1, 0])

    return p[1, 0]


def lr_gt_pearson():
    vocab = load_vocab("vocab/vocab_sst.pkl")
    model = torch.load(args.bow_snapshot, map_location="cpu")
    dict_path, label_path = (
        ".data/sst_raw/dictionary.txt",
        ".data/sst_raw/sentiment_labels.txt",
    )
    if BERT:
        phrase2id = load_txt_to_dict_hashed(dict_path)
    else:
        phrase2id = load_txt_to_dict(dict_path)
    id2label = load_txt_to_dict(label_path)

    a, b = [], []
    for word in vocab.stoi:
        if word in phrase2id:
            a.append(model.get_coefficient(vocab.stoi[word]))
            b.append(float(id2label[phrase2id[word]]))

    print(len(a))
    print(len(vocab.stoi))
    print(np.corrcoef(a, b)[1, 0])


def token2key(words):
    assert type(words) is list
    if len(words) > 4:
        return " ".join(words[:2] + words[-2:]), len(words)
    else:
        return " ".join(words), len(words)


def load_txt_to_dict_hashed(filename):
    f = open(filename)
    dic = {}
    for line in f.readlines():
        tup = line.lower().strip().split("|")
        if len(tup) != 2:
            continue
        key = tup[0].replace(" ", "")
        dic[key] = tup[1]
    return dic


def load_txt_to_dict(filename):
    f = open(filename)
    dic = {}
    for line in f.readlines():
        tup = line.lower().strip().split("|")
        if len(tup) != 2:
            continue
        dic[tup[0]] = tup[1]
    return dic


def phrase_gt_pearson(filename, dataset="dev"):
    f = open(filename)
    f2 = open(".data/sst/trees/%s.txt" % dataset)
    f3 = open("ground_truth.tmp", "w")
    dict_path, label_path = (
        ".data/sst_raw/dictionary.txt",
        ".data/sst_raw/sentiment_labels.txt",
    )
    if BERT:
        phrase2id = load_txt_to_dict_hashed(dict_path)
    else:
        phrase2id = load_txt_to_dict(dict_path)
    id2label = load_txt_to_dict(label_path)
    out, truth = [], []
    out_map, truth_map = {}, {}
    bucket_out, bucket_truth = {}, {}
    valid, total = 0, 0

    for idx, (line, line2) in enumerate(zip(f.readlines(), f2.readlines()[OFFSET:])):
        if idx < MINLINE:
            continue
        if idx == MAXLINE:
            break
        l = line.strip().split("\t")
        for entry in l:
            items = entry.strip().split(" ")
            score = float(items[-1])
            if BERT:
                key = "".join(items[:-1]).replace(" ", "").replace("##", "")
            else:
                key = " ".join(items[:-1])
            if key in phrase2id:
                phrase_id = phrase2id[key]
                gt_score = float(id2label[phrase_id])
                out.append(score)
                truth.append(gt_score)

                if phrase_id not in out_map:
                    out_map[phrase_id] = []
                out_map[phrase_id].append(score)
                truth_map[phrase_id] = gt_score
                bucket_key = len(items) - 1
                if bucket_key not in bucket_out:
                    bucket_out[bucket_key] = []
                    bucket_truth[bucket_key] = []
                bucket_out[bucket_key].append(score)
                bucket_truth[bucket_key].append(gt_score)

                valid += 1
            total += 1
    p = np.corrcoef(out, truth)
    print("phrase_corr", p[1, 0])

    e_out, e_truth = [], []
    for k in out_map:
        e_out.extend([np.mean(out_map[k])] * len(out_map[k]))
        e_truth.extend([truth_map[k]] * len(out_map[k]))
    p2 = np.corrcoef(e_out, e_truth)
    print("averaged phrase corr", p2[1, 0])

    print("(sanity check) matched: %d, total: %d" % (valid, total))

    return out, truth, p[1, 0], p2[1, 0]


def run_multiple(path):
    template = path
    nb_ranges, hists = [1, 2, 3, 4], [5, 10, 20]
    postfix = ".bert" if BERT else ""
    f = open(
        "analysis/%s_%s"
        % (TASK, template.split("/")[-1].replace("{", "").replace("}", "") + postfix),
        "w",
    )
    writer = csv.writer(f, delimiter="\t")
    for nb_range in nb_ranges:
        for h in hists:
            print(nb_range, h)
            path = template.format(**{"nb": nb_range, "h": h})
            if BERT and TASK == "sst":
                word_score = unigram_linear_pearson_bert_tree(path, dataset="test")
            elif TASK == "tacred":
                word_score = unigram_linear_pearson_multiclass(path)
            else:
                word_score = unigram_linear_pearson(path)
            if TASK == "sst":
                _, _, phrase_score, _ = phrase_gt_pearson(path, dataset="test")
            else:
                phrase_score = 0
            writer.writerow([nb_range, h, word_score, phrase_score])
    f.close()


if __name__ == "__main__":
    MINLINE = 0
    MAXLINE = 50
    OFFSET = 0
    DATASET = "test"
    path = args.eval_file
    BERT = "bert" in path
    TASK = ""

    for possible_task in ["sst", "yelp", "tacred"]:
        if possible_task in path:
            TASK = possible_task
            break
    VOCAB = None

    if TASK == "sst":
        args.bow_snapshot = get_best_snapshot("models/sst_bow")
        VOCAB = "vocab/vocab_sst.pkl"
    elif TASK == "yelp":
        args.bow_snapshot = get_best_snapshot("models/yelp_bow")
        VOCAB = "vocab/vocab_yelp.pkl"
    elif TASK == "tacred":
        args.bow_snapshot = get_best_snapshot("models/tacred_bow")
        VOCAB = "vocab/vocab_tacred.pkl"
    if BERT:
        if TASK == "sst":
            unigram_linear_pearson_bert_tree(path, DATASET)
        elif TASK == "yelp":
            unigram_linear_pearson(path)
        else:
            unigram_linear_pearson_multiclass(path)
    else:
        if TASK in ["sst", "yelp"]:
            unigram_linear_pearson(path)
        else:
            unigram_linear_pearson_multiclass(path)
    if TASK == "sst":
        out1, _, _, _ = phrase_gt_pearson(path, dataset=DATASET)
