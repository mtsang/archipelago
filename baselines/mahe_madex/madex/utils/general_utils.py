import torch
import numpy as np
from utils.lime import lime_base
import sklearn
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
import copy


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def force_float(X_numpy):
    return torch.from_numpy(X_numpy.astype(np.float32))


def proprocess_data(
    X, Y, valid_size=500, test_size=500, std_scale=False, std_scale_X=False
):

    n, p = X.shape
    ## Make dataset splits
    ntrain, nval, ntest = n - valid_size - test_size, valid_size, test_size

    Xs = {
        "train": X[:ntrain],
        "val": X[ntrain : ntrain + nval],
        "test": X[ntrain + nval : ntrain + nval + ntest],
    }
    Ys = {
        "train": np.expand_dims(Y[:ntrain], axis=1),
        "val": np.expand_dims(Y[ntrain : ntrain + nval], axis=1),
        "test": np.expand_dims(Y[ntrain + nval : ntrain + nval + ntest], axis=1),
    }

    for k in Xs:
        if len(Xs[k]) == 0:
            assert k != "train"
            del Xs[k]
            del Ys[k]

    if std_scale:
        scaler = StandardScaler()
        scaler.fit(Ys["train"])
        for k in Ys:
            Ys[k] = scaler.transform(Ys[k])
        Ys["scaler"] = scaler

    if std_scale_X:
        scaler = StandardScaler()
        scaler.fit(Xs["train"])
        for k in Xs:
            Xs[k] = scaler.transform(Xs[k])

    return Xs, Ys


class MLP(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_units,
        add_linear=False,
        act_func=nn.ReLU(),
    ):
        super(MLP, self).__init__()

        self.hidden_units = hidden_units
        self.add_linear = add_linear
        self.interaction_mlp = create_mlp(
            [num_features] + hidden_units + [1], act_func=act_func
        )

        self.add_linear = add_linear

        if add_linear:
            self.linear = nn.Linear(num_features, 1, bias=False)

    def forward(self, x):
        y = self.interaction_mlp(x)

        if self.add_linear:
            y += self.linear(x)
        return y


def create_mlp(layer_sizes, out_bias=True, act_func=nn.ReLU()):
    ls = list(layer_sizes)
    layers = nn.ModuleList()
    for i in range(1, len(ls) - 1):
        layers.append(nn.Linear(int(ls[i - 1]), int(ls[i])))
        layers.append(act_func)
    layers.append(nn.Linear(int(ls[-2]), int(ls[-1]), bias=out_bias))
    return nn.Sequential(*layers)


def train(
    net,
    data_loaders,
    criterion=nn.MSELoss(reduction="none"),
    nepochs=100,
    verbose=False,
    early_stopping=True,
    patience=5,
    l1_const=1e-4,
    l2_const=0,
    learning_rate=0.01,
    opt_func=optim.Adam,
    device=torch.device("cpu"),
    **kwargs
):
    optimizer = opt_func(net.parameters(), lr=learning_rate, weight_decay=l2_const)

    def include_sws(loss, sws):
        assert loss.shape == sws.shape
        return (loss * sws / sws.sum()).sum()

    def evaluate(net, data_loader, criterion, device):
        losses = []
        sws = []
        for inputs, targets, sws_batch in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss = criterion(net(inputs), targets).cpu().data
            losses.append(loss)
            sws.append(sws_batch)
        return include_sws(torch.stack(losses), torch.stack(sws)).item()

    best_loss = float("inf")
    best_net = None

    if "val" not in data_loaders:
        early_stopping = False

    patience_counter = 0

    for epoch in range(nepochs):
        if verbose:
            print("epoch", epoch)
        running_loss = 0.0
        run_count = 0
        for i, data in enumerate(data_loaders["train"], 0):
            inputs, targets, sws = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            sws = sws.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = include_sws(criterion(outputs, targets), sws)

            reg_loss = 0
            for name, param in net.named_parameters():
                if "interaction_mlp" in name and "weight" in name:
                    reg_loss += torch.sum(torch.abs(param))

            (loss + reg_loss * l1_const).backward()
            optimizer.step()
            running_loss += loss.item()
            run_count += 1

        if epoch % 1 == 0:
            key = "val" if "val" in data_loaders else "train"
            val_loss = evaluate(net, data_loaders[key], criterion, device)

            if verbose:
                print(
                    "[%d, %5d] train loss: %.4f, val loss: %.4f"
                    % (epoch + 1, nepochs, running_loss / run_count, val_loss)
                )
            if early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_net = copy.deepcopy(net)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > patience:
                        net = best_net
                        val_loss = best_loss
                        if verbose:
                            print("early stopping!")
                        break

            prev_loss = running_loss
            running_loss = 0.0

    if "test" in data_loaders:
        key = "test"
    elif "val" in data_loaders:
        key = "val"
    else:
        key = "train"
    test_loss = evaluate(net, data_loaders[key], criterion, device)

    if verbose:
        print("Finished Training. Test loss: ", test_loss)

    return net, test_loss


def merge_overlapping_sets(
    prediction_scores,
    interaction_atts,
    overlap_thresh=0.5,
    rel_gain_threshold=0,
    patience=1,
    num_features=None,
):
    def overlap_coef(A, B):
        A = set(A)
        B = set(B)
        return len(A & B) / min(len(A), len(B))

    def merge_sets(inter_sets):
        prev_sets = None
        inter_sets = list(inter_sets)
        inter_sets_merged = inter_sets
        while inter_sets != prev_sets:
            prev_sets = list(inter_sets)
            for A in inter_sets:
                for B in inter_sets_merged:
                    if A != B:
                        if overlap_coef(A, B) >= overlap_thresh:
                            inter_sets_merged.append(
                                tuple(sorted(set(A) | set(B)))
                            )  # merge
                            if A in inter_sets_merged:
                                inter_sets_merged.remove(A)
                            if B in inter_sets_merged:
                                inter_sets_merged.remove(B)

            inter_sets = list(set(inter_sets_merged))
        return inter_sets

    def threshold_inter_sets(interaction_atts, prediction_scores):
        scores = prediction_scores
        inter_sets = []
        patience_counter = 0
        best_score = scores[0]
        for i in range(1, len(scores)):
            cur_score = scores[i]
            rel_gain = (cur_score - best_score) / best_score
            inter_sets_temp, _ = zip(*interaction_atts[i - 1])
            if num_features is not None:
                if any(len(inter) == num_features for inter in inter_sets_temp):
                    break
            if rel_gain > rel_gain_threshold:
                best_score = cur_score
                inter_sets = inter_sets_temp
                patience_counter = 0
            else:
                if patience_counter < patience:
                    patience_counter += 1
                else:
                    break
        return inter_sets

    inter_sets = threshold_inter_sets(interaction_atts, prediction_scores)
    inter_sets_merged = merge_sets(inter_sets)

    return inter_sets_merged


######################################################
# The following are based on the official LIME repo
######################################################


def get_sample_distances(Xs):
    all_ones = np.ones((1, Xs["train"].shape[1]))
    Dd = {}
    for k in Xs:
        if k == "scaler":
            continue
        distances = sklearn.metrics.pairwise_distances(
            Xs[k], all_ones, metric="cosine"
        ).ravel()
        Dd[k] = distances

    return Dd


def get_sample_weights(Xs, kernel_width=0.25, enable=True, **kwargs):
    def kernel(d):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

    if enable:
        Dd = get_sample_distances(Xs)

    Wd = {}
    for k in Xs:
        if k == "scaler":
            continue
        if enable:
            Wd[k] = kernel(Dd[k])
        else:
            Wd[k] = np.ones(Xs[k].shape[0])

    return Wd


def get_lime_attributions(
    Xs, Ys, max_features=10000, kernel_width=0.25, weight_samples=True, sort=True
):
    def kernel(d):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

    distances = get_sample_distances(Xs)["train"]
    if not weight_samples:
        distances = np.ones_like(distances).squeeze(1)

    lb = lime_base.LimeBase(kernel_fn=kernel)
    lime_atts = lb.explain_instance_with_data(
        Xs["train"], Ys["train"], distances, 0, max_features
    )[0]
    if sort:
        lime_atts = sorted(lime_atts, key=lambda x: -x[1])
    return lime_atts
