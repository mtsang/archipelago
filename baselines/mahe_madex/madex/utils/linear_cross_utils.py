import torch
import numpy as np
import copy
from sklearn.linear_model import Ridge, base
from utils.general_utils import *


def update_cross_features(Xs_in, interactions):
    Xs = copy.deepcopy(Xs_in)

    for k in Xs:
        Xk = Xs[k]
        new_features = []

        for inter in interactions:
            inter_np = np.array(inter)
            new_feature = 1 * np.all(Xk[:, inter_np - 1], axis=1)
            new_features.append(new_feature)

        new_dset = np.concatenate([Xk, np.stack(new_features, axis=1)], axis=1)

        Xs[k] = new_dset
    return Xs


def fit_linear_cross_models(
    Xs,
    Ys,
    interactions,
    hierarchy_stepsize=1,
    max_steps=1,
    hierarchy_patience=0,
    stopping=False,
    verbose=False,
    weight_samples=False,
    flat=False,
    **kwargs
):
    """
    Trains an MLP and interprets interactions from its weights

    Args:
        data_loaders: dict of train, val, and test dataloaders
        sample_to_explain: the data instances to get attributions for
        interactions: a ranking of interaction sets
        hierarchy_stepsize: the stepsize across the ranking
        max_steps: the max number of steps on the ranking. a max_steps of 1 stops right after getting univariate attributions
        hierarchy patience: the patience of when to early stop on the interaction ranking based on validation performance
        user_linear: whether to use a linear model rather than a GAM for learning GAM+interactions model
        stopping: whether to early stop on the interaction ranking or not'
        mode: 'MSE' or 'BCE' for regression or binary classification
        experiment: name of experiment
        aggregate: aggregates the attributions of overlapping univariates and interaction sets
        verbose: set True to get training info

    Returns:
        the best GAM models, hierarchical interaction attributions, univariate attributions, prediction pefrformances at each hierarchical step, all trained GAMs
    """

    Wd = get_sample_weights(Xs, enable=weight_samples, **kwargs)

    best_model = None
    best_score = None
    margin = 0  # initialize to 0 to start, give initial slack for aggregate
    patience_counter = 0
    active_interaction_list = []
    hierarchical_interaction_attributions = []
    active_interactions = []
    prediction_scores = []

    break_out = False

    ## Build univariate gam
    n_features = Xs["train"].shape[1]  # next(iter(data_loaders["train"]))[0].shape[1]
    univariates = list(range(n_features))

    clf = Ridge(alpha=0.01)
    clf.fit(Xs["train"], Ys["train"], sample_weight=Wd["train"])
    r_sq = clf.score(Xs["val"], Ys["val"], sample_weight=Wd["val"])
    r_sq_test = clf.score(Xs["test"], Ys["test"], sample_weight=Wd["test"])

    Xs_base = copy.deepcopy(Xs)

    prediction_score = r_sq
    prediction_scores.append(r_sq_test)

    best_score = prediction_score

    univariate_attributions = (univariates, clf.coef_[0])

    for s in range(1, max_steps):
        active_interactions2 = []
        k = hierarchy_stepsize * s

        for v in range(k):
            try:
                interaction = interactions[v][0]
                active_interactions2.append(interaction)

            except:  # TODO handle this better later
                break_out = True
                break

            append, remove_items = True, []
            insertion_idx = len(active_interactions)
            for a, ai in enumerate(active_interactions):
                if set(interaction) <= set(ai):
                    append = False
                if set(interaction) > set(ai):
                    remove_items.append(ai)
                    if insertion_idx == len(active_interactions):
                        insertion_idx = a
            if remove_items:
                for r in remove_items:
                    active_interactions.remove(r)
            if append:
                active_interactions.insert(insertion_idx, interaction)

        if break_out:
            break

        active_interactions_pruned = [np.array(ai) for ai in active_interactions]
        active_interactions2 = active_interactions  # active_interactions_pruned

        if verbose:
            print("\tpruned", active_interactions_pruned)

        if flat:
            active_interactions2 = interactions

        Xs_inter = update_cross_features(Xs_base, active_interactions2)
        clf = Ridge(alpha=0.01)
        clf.fit(Xs_inter["train"], Ys["train"], sample_weight=Wd["train"])
        r_sq = clf.score(Xs_inter["val"], Ys["val"], sample_weight=Wd["val"])
        r_sq_test = clf.score(Xs_inter["test"], Ys["test"], sample_weight=Wd["test"])

        prediction_score = r_sq

        performance_improvement = prediction_score > best_score
        if (not stopping) or (
            stopping
            and (performance_improvement or patience_counter < hierarchy_patience)
        ):
            interaction_attributions = []
            for inter_i, inter in enumerate(active_interactions2):
                w = clf.coef_[0, inter_i + n_features]
                interaction_attributions.append((inter, w))
            hierarchical_interaction_attributions.append(interaction_attributions)
            prediction_scores.append(r_sq_test)

            if stopping:
                if performance_improvement:
                    patience_counter = 0
                    best_score = prediction_score
                else:
                    patience_counter += 1
        else:
            break

        if flat:
            return interaction_attributions, univariate_attributions, prediction_score

    return (
        prediction_scores,
        hierarchical_interaction_attributions,
        univariate_attributions,
    )
