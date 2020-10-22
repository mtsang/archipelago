from utils.pretrained.model_gcn import *
from collections import defaultdict
import numpy as np
import copy


def get_graph_model(model_folder):

    meta = torch.load(model_folder + "/gcn_cora.pt")

    n_hops = meta["n_hops"]
    n_nodes = meta["n_nodes"]
    test_idxs = meta["test_idxs"]
    n_samples = meta["n_samples"]
    dim_inp = meta["dim_inp"]
    dim_hid = meta["dim_hid"]
    dim_out = meta["dim_out"]

    model = create_model(dim_inp, dim_hid, dim_out, n_samples, n_hops)
    model.load_state_dict(meta["state_dict"])

    return model, n_nodes, n_hops, test_idxs


def convert_adj_to_da(adj_mat, make_undirected=False):
    # Converts adjacency to laplacian matrix
    if isinstance(adj_mat, np.ndarray):
        adj_mat = torch.from_numpy(adj_mat).float()
    if make_undirected:
        diag = torch.diag(torch.diag(adj_mat))
        x = adj_mat - diag
        adj_mat = x + x.t() + adj_mat

    da_mat = torch.eye(len(adj_mat)).to(adj_mat.device) - adj_mat
    return da_mat


def load_cora(data_folder, device):
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open(data_folder + "/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = [float(_) for _ in info[1:-1]]
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open(data_folder + "/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            n1 = node_map[info[0]]
            n2 = node_map[info[1]]
            adj_lists[n1].add(n2)
            adj_lists[n2].add(n1)

    adj_mat = np.zeros((num_nodes, num_nodes))
    for u in adj_lists:
        for v in adj_lists[u]:
            adj_mat[u, v] = 1

    feat_data = torch.FloatTensor(feat_data).to(device)
    adj_mat = torch.FloatTensor(adj_mat).to(device)
    return feat_data, adj_mat, labels


def get_hops_to_target(target_idx, adj_mat, n_hops):
    # Create a map from node to the number of hops from the target test index
    node_to_hop = {target_idx: 0}
    seen_points = {target_idx}
    for j in range(1, n_hops + 2):
        adj_cum = copy.deepcopy(adj_mat)
        for i in range(j - 1):
            adj_cum = torch.matmul(adj_cum, adj_mat)
        collect = {i for i, v in enumerate(adj_cum[target_idx]) if v != 0}
        ex_collect = collect - seen_points
        seen_points |= collect
        for e in ex_collect:
            node_to_hop[e] = j

    return node_to_hop
