import torch
import torch.nn as nn
from torch.nn.functional import relu


class InductiveGCN(nn.Module):
    def __init__(self, dim_inp, dim_hid, dim_out, n_samples, n_hops):
        super().__init__()

        self.dim_inp = dim_inp
        self.dim_hid = dim_hid
        self.dim_out = dim_out
        self.n_samples = n_samples

        dim_hiddens = [dim_inp] + [dim_hid] * n_hops
        self.layers = [
            nn.Linear(dim_hiddens[i], dim_hiddens[i + 1])
            for i in range(len(dim_hiddens) - 1)
        ]
        self.final_fc = nn.Linear(dim_hiddens[-1], dim_out)
        for layer in self.layers + [self.final_fc]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, adj_mat):
        """

        :param x: (n_nodes, dim_inp)
        :param adj_mat: (n_nodes, n_nodes)
        :return: (n_nodes, dim_out)
        """
        for layer in self.layers:
            x = torch.matmul(adj_mat, x)
            x = relu(layer(x))
        x = torch.matmul(adj_mat, x)
        x = self.final_fc(x)
        return x


def create_model(dim_inp, dim_hid, dim_out, n_samples, n_hops):
    return InductiveGCN(dim_inp, dim_hid, dim_out, n_samples, n_hops)
