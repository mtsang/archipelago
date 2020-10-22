import bisect
import operator
import numpy as np
import torch
from torch.utils import data
import torch.nn as nn
from utils.general_utils import *
from torch import autograd


def preprocess_weights(weights):
    w_later = np.abs(weights[-1])
    w_input = np.abs(weights[0])

    for i in range(len(weights) - 2, 0, -1):
        w_later = np.matmul(w_later, np.abs(weights[i]))

    return w_input, w_later


def interpret_interactions_from_weights(w_input, w_later, get_main_effects=False):
    interaction_strengths = {}
    for i in range(w_later.shape[1]):
        sorted_hweights = sorted(
            enumerate(w_input[i]), key=lambda x: x[1], reverse=True
        )
        interaction_candidate = []
        candidate_weights = []
        for j in range(w_input.shape[1]):
            bisect.insort(interaction_candidate, sorted_hweights[j][0])
            candidate_weights.append(sorted_hweights[j][1])

            if not get_main_effects and len(interaction_candidate) == 1:
                continue
            interaction_tup = tuple(interaction_candidate)
            if interaction_tup not in interaction_strengths:
                interaction_strengths[interaction_tup] = 0
            interaction_strength = (min(candidate_weights)) * (np.sum(w_later[:, i]))
            interaction_strengths[interaction_tup] += interaction_strength

    interaction_ranking = sorted(
        interaction_strengths.items(), key=operator.itemgetter(1), reverse=True
    )

    return interaction_ranking


def interpret_pairwise_interactions(w_input, w_later):
    p = w_input.shape[1]

    interaction_ranking = []
    for i in range(p):
        for j in range(p):
            if i < j:
                strength = (np.minimum(w_input[:, i], w_input[:, j]) * w_later).sum()
                interaction_ranking.append(((i, j), strength))

    interaction_ranking.sort(key=lambda x: x[1], reverse=True)
    return interaction_ranking


def get_higher_order_grad(inter, model, x, device):
    x = torch.FloatTensor(x).to(device)
    x.requires_grad = True
    y = model(x)
    for i, v in enumerate(inter):
        if i == 0:
            grad = autograd.grad(y, x, create_graph=True)[0][v]  # first feature
        else:
            grad = autograd.grad(grad, x, create_graph=True)[0][v]  # second feature

    return grad.item() ** 2


def get_second_order_grad(model, x, device):

    x = torch.FloatTensor(x).to(device)

    if x.nelement() < 2:
        return np.array([])

    x.requires_grad = True

    y = model(x)
    grads = autograd.grad(y, x, create_graph=True)[0].squeeze()

    grad_list = []
    for j, grad in enumerate(grads):
        grad2 = autograd.grad(grad, x, retain_graph=True)[0].squeeze()
        grad_list.append(grad2)

    grad_matrix = torch.stack(grad_list)
    return grad_matrix.cpu().numpy() ** 2


def run_NID(weights, pairwise=False):
    w_input, w_later = preprocess_weights(weights)

    if pairwise:
        interaction_ranking_pruned = interpret_pairwise_interactions(w_input, w_later)
    else:
        interaction_ranking = interpret_interactions_from_weights(w_input, w_later)
        interaction_ranking_pruned = prune_redundant_interactions(interaction_ranking)

    return interaction_ranking_pruned


def run_gradient_NID(mlp, x, grad_gpu):
    interaction_scores = {}

    if grad_gpu == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(grad_gpu))

    mlp = mlp.to(device)

    inter_matrix = get_second_order_grad(mlp, x, device)

    if len(inter_matrix) == 0:
        return []

    inter_scores = []

    for j in range(inter_matrix.shape[0]):
        for i in range(j):
            inter_scores.append(((i, j), inter_matrix[i, j]))

    inter_ranking = sorted(inter_scores, key=lambda x: -x[1])

    return inter_ranking


def prune_redundant_interactions(interaction_ranking, max_interactions=100):
    interaction_ranking_pruned = []
    current_superset_inters = []
    for inter, strength in interaction_ranking:
        set_inter = set(inter)
        if len(interaction_ranking_pruned) >= max_interactions:
            break
        subset_inter_skip = False
        update_superset_inters = []
        for superset_inter in current_superset_inters:
            if set_inter < superset_inter:
                subset_inter_skip = True
                break
            elif not (set_inter > superset_inter):
                update_superset_inters.append(superset_inter)
        if subset_inter_skip:
            continue
        current_superset_inters = update_superset_inters
        current_superset_inters.append(set_inter)
        interaction_ranking_pruned.append((inter, strength))

    return interaction_ranking_pruned


def detect_interactions(
    Xs,
    Ys,
    detector="NID",
    x_instance_representation=None,
    arch=[256, 128, 64],
    batch_size=100,
    device=torch.device("cpu"),
    weight_samples=False,
    add_linear=False,
    l1_const=None,
    grad_gpu=-1,
    seed=None,
    pairwise=False,
    **kwargs
):
    def get_weights(model):
        weights = []
        for name, param in model.named_parameters():
            if "interaction_mlp" in name and "weight" in name:
                weights.append(param.cpu().detach().numpy())
        return weights

    assert detector in {"NID", "GradientNID"}

    if seed is not None:
        set_seed(seed)

    if type(Xs) != dict and type(Ys) != dict:
        Xs = {"train": Xs}
        Ys = {"train": Ys}

    Wd = get_sample_weights(Xs, enable=weight_samples, **kwargs)

    data_loaders = {}
    for k in Xs:
        feats = force_float(Xs[k])
        targets = force_float(Ys[k])
        sws = force_float(Wd[k]).unsqueeze(1)
        dataset = data.TensorDataset(feats, targets, sws)
        data_loaders[k] = data.DataLoader(dataset, batch_size)

    if detector == "GradientNID":
        act_func = nn.Softplus()
        if l1_const == None:
            l1_const = 0
    else:
        act_func = nn.ReLU()
        if l1_const == None:
            l1_const = 1e-4

    mlp = MLP(feats.shape[1], arch, add_linear=add_linear, act_func=act_func).to(device)

    mlp, mlp_loss = train(mlp, data_loaders, device=device, l1_const=l1_const, **kwargs)

    if detector == "NID":
        inters = run_NID(get_weights(mlp), pairwise)
    elif detector == "GradientNID":
        if x_instance_representation is None:
            x_instance_representation = np.ones((1, Xs["train"].shape[1]))
        inters = run_gradient_NID(mlp, x_instance_representation, grad_gpu)

    return inters, mlp_loss
