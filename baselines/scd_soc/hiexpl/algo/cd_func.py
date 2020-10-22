# MIT License
#
# Copyright (c) 2018 William James Murdoch
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


import os
import pdb
import torch
import numpy as np
from torchtext import data, datasets
import random
from math import e
from utils.args import args


def sigmoid(x):
    return 1 / (1 + e ** (-x))


def tanh(x):
    return (1 - e ** (-2 * x)) / (1 + e ** (-2 * x))


def get_model(snapshot_file):
    print("loading", snapshot_file)
    try:  # load onto gpu
        model = torch.load(snapshot_file)
        print("loaded onto gpu...")
    except:  # load onto cpu
        model = torch.load(snapshot_file, map_location=lambda storage, loc: storage)
        print("loaded onto cpu...")
    return model


def get_sst():
    inputs = data.Field(lower="preserve-case")
    answers = data.Field(sequential=False, unk_token=None)

    # build with subtrees so inputs are right
    train_s, dev_s, test_s = datasets.SST.splits(
        inputs,
        answers,
        fine_grained=False,
        train_subtrees=True,
        filter_pred=lambda ex: ex.label != "neutral",
    )
    inputs.build_vocab(train_s, dev_s, test_s)
    answers.build_vocab(train_s)

    # rebuild without subtrees to get longer sentences
    train, dev, test = datasets.SST.splits(
        inputs,
        answers,
        fine_grained=False,
        train_subtrees=False,
        filter_pred=lambda ex: ex.label != "neutral",
    )

    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=1, device=0
    )

    return inputs, answers, train_iter, dev_iter


def get_batches(batch_nums, train_iterator, dev_iterator, dset="train"):
    print("getting batches...")
    np.random.seed(0)
    random.seed(0)

    # pick data_iterator
    if dset == "train":
        data_iterator = train_iterator
    elif dset == "dev":
        data_iterator = dev_iterator

    # actually get batches
    num = 0
    batches = {}
    data_iterator.init_epoch()
    for batch_idx, batch in enumerate(data_iterator):
        if batch_idx == batch_nums[num]:
            batches[batch_idx] = batch
            num += 1

        if num == max(batch_nums):
            break
        elif num == len(batch_nums):
            print("found them all")
            break
    return batches


def evaluate_predictions(snapshot_file):
    print("loading", snapshot_file)
    try:  # load onto gpu
        model = torch.load(snapshot_file)
    except:  # load onto cpu
        model = torch.load(snapshot_file, map_location=lambda storage, loc: storage)
    inputs = data.Field()
    answers = data.Field(sequential=False, unk_token=None)

    train, dev, test = datasets.SST.splits(
        inputs,
        answers,
        fine_grained=False,
        train_subtrees=False,
        filter_pred=lambda ex: ex.label != "neutral",
    )
    inputs.build_vocab(train)
    answers.build_vocab(train)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=1, device=0
    )
    train_iter.init_epoch()
    for batch_idx, batch in enumerate(train_iter):
        print("batch_idx", batch_idx)
        out = model(batch)
        target = batch.label
        break
    return batch, out, target


def is_in_intervals(idx, intervals):
    for interval in intervals:
        if interval[0] <= idx <= interval[1]:
            return True
    return False


def partition_bias(rel, irrel, bias):
    b_r = bias * abs(rel) / (abs(rel) + abs(irrel) + 1e-12)
    b_ir = bias * (abs(irrel) + 1e-12) / (abs(rel) + abs(irrel) + 1e-12)
    return b_r, b_ir


def CD(batch, model, intervals):
    if not args.task == "tacred":
        word_vecs = model.embed(batch.text)[:, 0]
        lstm_module = model.lstm
    else:
        token_vec = model.embed(batch.text)
        pos_vec = model.pos_embed(batch.pos)
        ner_vec = model.ner_embed(batch.ner)
        word_vecs = model.drop(torch.cat([token_vec, pos_vec, ner_vec], -1))[:, 0]
        lstm_module = model.lstm.rnn

    weights = lstm_module.state_dict()

    # Index one = word vector (i) or hidden state (h), index two = gate
    W_ii, W_if, W_ig, W_io = np.split(weights["weight_ih_l0"], 4, 0)
    W_hi, W_hf, W_hg, W_ho = np.split(weights["weight_hh_l0"], 4, 0)
    b_i, b_f, b_g, b_o = np.split(
        weights["bias_ih_l0"].cpu().numpy() + weights["bias_hh_l0"].cpu().numpy(), 4
    )
    # word_vecs = model.embed(batch.text)[:, 0].data
    T = word_vecs.size(0)
    relevant = np.zeros((T, model.hidden_dim))
    irrelevant = np.zeros((T, model.hidden_dim))
    relevant_h = np.zeros((T, model.hidden_dim))
    irrelevant_h = np.zeros((T, model.hidden_dim))
    for i in range(T):
        if i > 0:
            prev_rel_h = relevant_h[i - 1]
            prev_irrel_h = irrelevant_h[i - 1]
        else:
            prev_rel_h = np.zeros(model.hidden_dim)
            prev_irrel_h = np.zeros(model.hidden_dim)

        rel_i = np.dot(W_hi, prev_rel_h)
        rel_g = np.dot(W_hg, prev_rel_h)
        rel_f = np.dot(W_hf, prev_rel_h)
        rel_o = np.dot(W_ho, prev_rel_h)
        irrel_i = np.dot(W_hi, prev_irrel_h)
        irrel_g = np.dot(W_hg, prev_irrel_h)
        irrel_f = np.dot(W_hf, prev_irrel_h)
        irrel_o = np.dot(W_ho, prev_irrel_h)

        if is_in_intervals(i, intervals):
            rel_i = rel_i + np.dot(W_ii, word_vecs[i])
            rel_g = rel_g + np.dot(W_ig, word_vecs[i])
            rel_f = rel_f + np.dot(W_if, word_vecs[i])
            rel_o = rel_o + np.dot(W_io, word_vecs[i])
        else:
            irrel_i = irrel_i + np.dot(W_ii, word_vecs[i])
            irrel_g = irrel_g + np.dot(W_ig, word_vecs[i])
            irrel_f = irrel_f + np.dot(W_if, word_vecs[i])
            irrel_o = irrel_o + np.dot(W_io, word_vecs[i])

        rel_contrib_i, irrel_contrib_i, bias_contrib_i = decomp_three(
            rel_i, irrel_i, b_i, sigmoid
        )
        rel_contrib_g, irrel_contrib_g, bias_contrib_g = decomp_three(
            rel_g, irrel_g, b_g, np.tanh
        )

        # i_value = sum([rel_contrib_i, irrel_contrib_i, bias_contrib_i])
        # g_value = sum([rel_contrib_g, irrel_contrib_g, bias_contrib_g])
        # print(i_value[:10], g_value[:10])

        relevant[i] = (
            rel_contrib_i * (rel_contrib_g + bias_contrib_g)
            + bias_contrib_i * rel_contrib_g
        )
        irrelevant[i] = (
            irrel_contrib_i * (rel_contrib_g + irrel_contrib_g + bias_contrib_g)
            + (rel_contrib_i + bias_contrib_i) * irrel_contrib_g
        )

        if is_in_intervals(i, intervals):
            relevant[i] += bias_contrib_i * bias_contrib_g
        else:
            irrelevant[i] += bias_contrib_i * bias_contrib_g

        # c = relevant[i] + irrelevant[i]
        # print('_', c[:10])

        if i > 0:
            rel_contrib_f, irrel_contrib_f, bias_contrib_f = decomp_three(
                rel_f, irrel_f, b_f, sigmoid
            )
            relevant[i] += (rel_contrib_f + bias_contrib_f) * relevant[i - 1]
            irrelevant[i] += (
                rel_contrib_f + irrel_contrib_f + bias_contrib_f
            ) * irrelevant[i - 1] + irrel_contrib_f * relevant[i - 1]

        # c = relevant[i] + irrelevant[i]
        # print(c[:10])

        o = sigmoid(
            np.dot(W_io, word_vecs[i]) + np.dot(W_ho, prev_rel_h + prev_irrel_h) + b_o
        )
        # print('o', o[:10])
        # rel_contrib_o, irrel_contrib_o, bias_contrib_o = decomp_three(rel_o, irrel_o, b_o, sigmoid)
        new_rel_h, new_irrel_h = decomp_tanh_two(relevant[i], irrelevant[i])
        # h = new_rel_h + new_irrel_h
        # print(h[:10])
        # relevant_h[i] = new_rel_h * (rel_contrib_o + bias_contrib_o)
        # irrelevant_h[i] = new_rel_h * irrel_contrib_o + new_irrel_h * (rel_contrib_o + irrel_contrib_o + bias_contrib_o)
        relevant_h[i] = o * new_rel_h
        irrelevant_h[i] = o * new_irrel_h

    W_out = model.hidden_to_label.weight.data

    if args.task == "tacred":
        relevant_h[T - 1] = (
            model.drop(torch.from_numpy(relevant_h[T - 1]).view(1, -1).cuda())
            .view(-1)
            .cpu()
            .numpy()
        )
    # Sanity check: scores + irrel_scores should equal the LSTM's output minus model.hidden_to_label.bias
    if not args.mean_hidden:
        scores = np.dot(W_out, relevant_h[T - 1])
        irrel_scores = np.dot(W_out, irrelevant_h[T - 1])
    else:
        scores = np.dot(W_out, np.mean(relevant_h, 0))
        irrel_scores = np.dot(W_out, np.mean(irrelevant_h[T - 1]))

    return scores, irrel_scores


def decomp_three(a, b, c, activation):
    a_contrib = 0.5 * (activation(a + c) - activation(c)) + 0.5 * (
        activation(a + b + c) - activation(b + c)
    )
    c_contrib = activation(c)
    b_contrib = activation(a + b + c) - a_contrib - c_contrib
    return a_contrib, b_contrib, c_contrib


def decomp_tanh_two(a, b):
    return 0.5 * (np.tanh(a) + (np.tanh(a + b) - np.tanh(b))), 0.5 * (
        np.tanh(b) + (np.tanh(a + b) - np.tanh(a))
    )


def decomp_activation_two_pad(a, b, activation):
    a_contrib = activation(a)
    b_contrib = activation(a + b) - a_contrib
    return a_contrib, b_contrib


def decomp_three_pad(a, b, c, activation):
    a_contrib = 1 / 2 * (activation(a + c) - activation(c)) + 1 / 2 * (
        activation(a + b + c) - activation(b + c)
    )
    c_contrib = activation(c)
    b_contrib = activation(a + b + c) - a_contrib - c_contrib
    return a_contrib, b_contrib, c_contrib
