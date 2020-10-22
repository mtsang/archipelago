import numpy as np
from application_utils.common_utils import get_efficient_mask_indices
import pickle
import copy


class TextXformer:
    # note: this xformer is not the transformer from Vaswani et al., 2017

    def __init__(self, input_ids, baseline_ids):
        self.input = input_ids
        self.baseline = baseline_ids
        self.num_features = len(self.input)

    def simple_xform(self, inst):
        mask_indices = np.argwhere(inst == True).flatten()
        id_list = list(self.baseline)
        for i in mask_indices:
            id_list[i] = self.input[i]
        return id_list

    def efficient_xform(self, inst):
        mask_indices, base, change = get_efficient_mask_indices(
            inst, self.baseline, self.input
        )
        for i in mask_indices:
            base[i] = change[i]
        return base

    def __call__(self, inst):
        id_list = self.efficient_xform(inst)
        return id_list


def process_stop_words(explanation, tokens, strip_first_last=True):
    explanation = copy.deepcopy(explanation)
    tokens = copy.deepcopy(tokens)
    stop_words = set(
        [
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "were",
            "will",
            "with",
            "s",
            "ll",
        ]
    )
    for i, token in enumerate(tokens):
        if token in stop_words:
            if (i,) in explanation:
                explanation[(i,)] = 0.0

    if strip_first_last:
        explanation.pop((0,))
        explanation.pop((len(tokens) - 1,))
        tokens = tokens[1:-1]
    return explanation, tokens


def get_input_baseline_ids(text, baseline_token, tokenizer):
    input_ids = prepare_huggingface_data([text, baseline_token], tokenizer)["input_ids"]
    text_ids = input_ids[0]
    baseline_ids = np.array(
        [input_ids[0][0]] + [input_ids[1, 1]] * (len(text_ids) - 2) + [input_ids[0][-1]]
    )
    return text_ids, baseline_ids


def get_token_list(sentence, tokenizer):
    if isinstance(sentence, str):
        X = prepare_huggingface_data([sentence], tokenizer)
        batch_ids = X["input_ids"]
    else:
        batch_ids = np.expand_dims(sentence, 0)
    token_list = []
    for i in range(batch_ids.shape[0]):
        ids = batch_ids[i]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        token_list.append(tokens)
    return token_list[0]


def get_sst_sentences(split="test", path="../../downloads/sst_data/sst_trees.pickle"):
    with open(path, "rb") as handle:
        sst_trees = pickle.load(handle)

    data = []
    for s in range(len(sst_trees[split])):
        sst_item = sst_trees[split][s]

        sst_item = sst_trees[split][s]
        sentence = sst_item[0]
        data.append(sentence)
    return data


def prepare_huggingface_data(sentences, tokenizer):
    X = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
    for sentence in sentences:
        encoded_sentence = tokenizer.encode_plus(sentence, add_special_tokens=True)
        for key in encoded_sentence:
            X[key].append(encoded_sentence[key])

        assert not any(encoded_sentence["token_type_ids"])

    # pad to the batch max length (auto-identified from encode_plus)
    batch_ids = X["input_ids"]
    max_len = np.max([len(ids) for ids in batch_ids])
    X_pad = {}
    for i, ids in enumerate(batch_ids):
        diff = max_len - len(ids)
        for key in X:
            if key not in X_pad:
                X_pad[key] = []
            X_pad[key].append(X[key][i] + [0] * diff)

    for key in X_pad:
        X_pad[key] = np.array(X_pad[key])
    return X_pad
