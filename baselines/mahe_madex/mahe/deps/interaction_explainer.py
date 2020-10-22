import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data_utils
import torch.optim as optim
from torch.nn.parameter import Parameter
from functools import reduce
import operator as op
import copy
from utils.general_utils import train, get_sample_weights, set_seed, force_float


def include_sws(loss, sws):
    assert loss.shape == sws.shape
    return (loss * sws / sws.sum()).sum()


def evaluate_net(
    net,
    data_loader,
    device,
    criterion=nn.MSELoss(reduction="none"),
    printout="",
    display=True,
):
    losses = []
    sws = []
    for inputs, targets, sws_batch in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        loss = criterion(net(inputs), targets).cpu().data
        losses.append(loss)
        sws.append(sws_batch)
    return include_sws(torch.stack(losses), torch.stack(sws)).item()


def create_MLP2(n_inp, n_out, hidden_units, activation=nn.ReLU, linear_bias=True):
    layers = []
    layers_size = [n_inp] + hidden_units
    for i in range(len(layers_size) - 1):
        layers.append(nn.Linear(layers_size[i], layers_size[i + 1]))
        if activation is not None:
            layers.append(activation())
    layers.append(nn.Linear(layers_size[-1], n_out, bias=linear_bias))

    return nn.Sequential(*layers)


class MLP2(nn.Module):
    def __init__(self, n_inp, n_out, hidden_units, **kwargs):
        super(MLP, self).__init__()
        self.mlp = create_MLP2(n_inp, n_out, hidden_units)

    def forward(self, x):
        return self.mlp(x)


class Additive_MLP(nn.Module):
    def __init__(
        self,
        n_out,
        hidden_units,
        interactions,
        univariates,
        out_bias=True,
        linear=True,
        **kwargs
    ):
        super(Additive_MLP, self).__init__()

        self.interactions = interactions
        self.univariates = [[u] for u in univariates]
        self.use_linear = linear
        self.n_out, self.hidden_units, self.out_bias = n_out, hidden_units, out_bias
        self.interaction_mlps = self.create_additive_MLP(
            interactions, n_out, hidden_units, False, "inter"
        )
        if self.use_linear:
            self.linear = nn.Linear(len(self.univariates), n_out, bias=False)
        else:
            self.univariate_mlps = self.create_additive_MLP(
                self.univariates, n_out, hidden_units, False, "uni"
            )
        self.inter_coef = 1
        self.inter_coef_react = []
        self.bias = Parameter(force_float(np.array([0.0])))

    def forward(self, x):
        y = self.forward_additive_MLP(
            x,
            self.interactions,
            self.interaction_mlps,
            inter_coef_react=self.inter_coef_react,
        )

        if self.use_linear:
            y += self.linear(x)
        else:
            y += self.forward_additive_MLP(
                x,
                self.univariates,
                self.univariate_mlps,
                inter_coef_react=self.inter_coef_react,
            )
        if self.out_bias:
            y += self.bias
        return y

    def create_additive_MLP(self, input_groupings, n_out, hidden_units, out_bias, name):
        mlp_list = [
            create_MLP2(len(grouping), n_out, hidden_units, linear_bias=out_bias)
            for grouping in input_groupings
        ]
        for i in range(len(input_groupings)):
            setattr(self, "_" + name + "_" + str(i), mlp_list[i])
        return mlp_list

    def forward_additive_MLP(self, x, input_groupings, mlps, inter_coef_react=[]):
        forwarded_mlps = []
        for i, mlp in enumerate(mlps):
            grouping = np.array(input_groupings[i])
            if i in inter_coef_react:
                forwarded_mlps.append(self.inter_coef * mlp(x[:, grouping]))
                print("negated mlp is:", mlp)
                print("negated by:", self.inter_coef)
            # grouping2 = np.array([g - 1 for g in grouping])
            else:
                forwarded_mlps.append(mlp(x[:, grouping]))
        forwarded_mlp = sum(forwarded_mlps)
        return forwarded_mlp

    def freeze_univariates(self):
        self.frozen_params = True
        for name, param in self.named_parameters():
            if name is not "bias":
                if name.startswith("_uni_") or not name.startswith("_inter_"):
                    param.requires_grad = False  # Freeze

    def update_interactions(self, interactions, hidden_units=[], shift=False):
        if shift:
            self.interactions = []
            for inter in interactions:
                self.interactions.append(inter - 1)
        else:
            self.interactions = interactions
        inter_hidden_units = hidden_units if hidden_units else self.hidden_units
        # self.bias.data.fill_(0.0)
        # print(self.interactions, inter_hidden_units)

        self.interaction_mlps = self.create_additive_MLP(
            self.interactions, self.n_out, inter_hidden_units, False, "inter"
        )  # .to(device)


# TODO incorporate x by mul by w in linear


def get_importance_scores(
    data_loaders,
    interactions,
    hierarchy_stepsize,
    truncation_mode,
    num_steps,
    device,
    verbose=False,
    out_bias=True,
    use_linear=True,
    stopping=True,
    hierarchical_patience=0,
    seed=None,
    skip_end=False,
    **kwargs
):
    if seed is not None:
        set_seed(seed)

    X_dummy, _, _ = next(iter(data_loaders["val"]))
    n_features = X_dummy.shape[1]
    sample_to_explain = torch.FloatTensor(np.ones((1, n_features))).to(device)
    # n_features = test_data[0].shape[1]
    univariates = np.array(range(0, n_features))
    best_model = None
    best_score = None
    margin = 0
    patience_counter = 0
    active_interaction_list = []

    hierarchical_interaction_contributions = []
    # hierarchical_linear_contributions = []
    active_interactions = []
    prediction_scores = []
    prediction_scores2 = []

    break_out = False

    gams = []
    val_losses = []

    ################################
    #    build univariate gam
    ################################

    univariates = list(range(n_features))

    model = Additive_MLP(
        1,
        [10, 10, 10],
        [],
        list(range(n_features)),
        linear=use_linear,
    ).to(device)

    if use_linear:
        learning_rate = 1e-1
        l1_const = 0  # 1e-5
        weight_decay = 1e-4  # 1e-1
        n_epochs = 20
    else:
        learning_rate = 1e-3
        l1_const = 0  # 1e-6
        weight_decay = 1e-5
        n_epochs = 5

    #     model = train_net(model, train_loader, test_loader, l1_const = l1_const, verbose = verbose, mode=mode, learning_rate=learning_rate, epochs=n_epochs, weight_decay=weight_decay, early_stopping=True, val_loader=val_loader)
    model, test_loss = train(
        model,
        data_loaders,
        verbose=verbose,
        l1_const=0,
        l2_const=1e-5,
        learning_rate=1e-2,
        device=device,
        **kwargs
    )

    prediction_scores.append(test_loss)

    val_loss = evaluate_net(model, data_loaders["val"], device, printout="step_0")

    val_losses.append(val_loss)

    if verbose:
        print("step", "0", val_loss, test_loss)

    gams.append(copy.deepcopy(model))

    if stopping:
        best_score = val_loss
        best_model = copy.deepcopy(model)

    if use_linear:
        univariate_contributions = [
            (univariates, param2.cpu().data.numpy())
            for param2 in [param for param in model.linear.parameters()][0]
        ][0]
    else:
        univariate_contributions = []
        for uni in range(n_features):
            univariate_contribution = model.univariate_mlps[uni](
                sample_to_explain[:, uni]
            ).item()
            univariate_contributions.append((uni, univariate_contribution))

    ################################
    #    build interaction gam
    ################################

    model.freeze_univariates()

    for s in range(1, num_steps):
        i = hierarchy_stepsize * (s - 1)
        if skip_end:
            i = len(interactions) - 1
        if s - 1 >= 0:

            for v in range(i + 1):
                try:
                    interaction = interactions[v][0]
                except:  # TODO handle this better later
                    break_out = True
                    # print('no more interactions left')
                    break
                if interaction not in active_interactions:
                    active_interactions.append(interaction)

            if break_out:
                break

            active_interactions_pruned = []

            if active_interactions:
                for inter1 in active_interactions:
                    skip = False
                    for inter2 in active_interactions:
                        if set(inter1) < set(inter2):
                            skip = True
                            break
                    if not skip:
                        active_interactions_pruned.append(np.array(inter1))
        else:
            active_interactions_pruned = []

        if verbose:
            print(active_interactions_pruned)
        #         model.update_interactions(active_interactions_pruned, hidden_units = [30,15], shift=False)
        model.update_interactions(
            active_interactions_pruned, hidden_units=[64, 32, 16], shift=False
        )
        model = model.to(device)

        # learning_rate->1e-3, decay=1e-3
        model, test_loss = train(
            model,
            data_loaders,
            verbose=verbose,
            l1_const=0,
            l2_const=1e-5,
            learning_rate=1e-3,
            device=device,
            **kwargs
        )

        prediction_score = test_loss

        #         model = train_net(model, train_loader, test_loader, l1_const = 0, learning_rate=1e-3, weight_decay=1e-5, verbose = verbose, mode=mode, early_stopping=(val_loader is not None), val_loader=val_loader, epochs=20)

        val_loss = evaluate_net(
            model, data_loaders["val"], device, printout="step_" + str(s)
        )
        val_losses.append(val_loss)

        if verbose:
            print("step", s, val_loss, test_loss)

        if stopping:
            if (val_loss < best_score) or (patience_counter < hierarchical_patience):

                best_model = copy.deepcopy(model)
                gams.append(copy.deepcopy(model))

                interaction_contributions = []
                for inter_i, inter in enumerate(active_interactions_pruned):
                    interaction_contribution = model.interaction_mlps[inter_i](
                        sample_to_explain[:, inter]
                    ).item()
                    interaction_contributions.append((inter, interaction_contribution))

                hierarchical_interaction_contributions.append(interaction_contributions)
                prediction_scores.append(prediction_score)

                if val_loss < best_score:
                    best_score = val_loss
                    patience_counter = 0

                elif patience_counter < hierarchical_patience:
                    patience_counter += 1

            else:
                #                 if patience_counter >= patience:
                print("stop training and using the previous model")
                assert best_model is not None
                break

        else:
            gams.append(copy.deepcopy(model))

            interaction_contributions = []
            for inter_i, inter in enumerate(active_interactions_pruned):
                interaction_contribution = model.interaction_mlps[inter_i](
                    sample_to_explain[:, inter]
                ).item()
                interaction_contributions.append((inter, interaction_contribution))

            hierarchical_interaction_contributions.append(interaction_contributions)
            prediction_scores.append(prediction_score)

        if skip_end:
            break

    if stopping:
        model = best_model

    for i, val_loss in enumerate(val_losses):
        if val_loss == best_score:
            break

    hierarchical_interaction_contributions = hierarchical_interaction_contributions[:i]
    prediction_scores = prediction_scores[: i + 1]

    return (
        model,
        hierarchical_interaction_contributions,
        univariate_contributions,
        prediction_scores,
    )


def learn_hierarchical_gam(
    Xd,
    Yd,
    interactions,
    device,
    verbose=False,
    hierarchy_stepsize=1,
    truncation_mode="fixed_num_steps",
    num_steps=1,
    weight_samples=None,
    out_bias=True,
    batch_size=100,
    get_gams=False,
    use_linear=True,
    stopping=True,
    hierarchical_patience=0,
    skip_end=False,
    **kwargs
):

    Wd = get_sample_weights(Xd, enable=weight_samples, **kwargs)

    data_loaders = {}
    for k in Xd:
        feats = force_float(Xd[k])  # torch.FloatTensor(Xd[k])#.to(device)
        labels = force_float(Yd[k])  # torch.FloatTensor(Yd[k])#.to(device)
        sws = force_float(Wd[k]).unsqueeze(1)
        dataset = data_utils.TensorDataset(feats, labels, sws)
        data_loaders[k] = data_utils.DataLoader(dataset, batch_size)

    (
        net,
        interaction_contributions,
        univariate_contributions,
        prediction_scores,
    ) = get_importance_scores(
        data_loaders,
        interactions,
        hierarchy_stepsize,
        truncation_mode,
        num_steps,
        device,
        verbose=verbose,
        out_bias=out_bias,
        use_linear=use_linear,
        stopping=stopping,
        hierarchical_patience=hierarchical_patience,
        skip_end=skip_end,
        **kwargs
    )

    #     net, interaction_contributions, univariate_contributions, prediction_scores = None, None, None, None

    return interaction_contributions, univariate_contributions, prediction_scores
