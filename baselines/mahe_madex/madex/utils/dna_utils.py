import torch.nn as nn
import torch
import h5py as h5
import numpy as np
from utils.general_utils import *

# from sampling_and_inference import *


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def createConv1D(n_inp, n_out, hidden_units, kernel_size, seq_len, activation=nn.ReLU):

    layers = []
    layers_size = [n_inp] + hidden_units
    for i in range(len(layers_size) - 1):
        layers.append(nn.Conv1d(layers_size[i], layers_size[i + 1], kernel_size))
        if activation is not None:
            layers.append(activation())
    layers.append(Flatten())
    seq_len = seq_len - (kernel_size - 1) * len(hidden_units)
    linear_dim = layers_size[-1] * seq_len
    layers.append(nn.Linear(linear_dim, n_out))

    return nn.Sequential(*layers)


class conv1D(nn.Module):
    def __init__(self, n_inp, n_out, hidden_units, kernel_size, seq_len, **kwargs):
        super(conv1D, self).__init__()
        self.conv1D = createConv1D(n_inp, n_out, hidden_units, kernel_size, seq_len)

    def forward(self, x):
        return self.conv1D(x)


def load_dna_model(path):
    model = conv1D(4, 1, [64, 64], 5, 36)
    model.load_state_dict(torch.load(path))
    return model


def generate_random_dna_sequence_with_CACGTG(length=36, seed=None):
    if seed is not None:
        set_seed(seed)

    nucleotides = ["A", "C", "G", "T"]
    seq = ""
    ebox = "CACGTG"
    for i in np.random.randint(0, 4, (length)):
        seq += nucleotides[i]
    i = np.random.randint(0, length - len(ebox))
    seq = seq[:i] + ebox + seq[i + len(ebox) :]
    return seq


def encode_dna_onehot(seq):
    seq_as_list = list(seq)

    for i, c in enumerate(seq_as_list):
        if c == "A":
            seq_as_list[i] = [1, 0, 0, 0]
        elif c == "T":
            seq_as_list[i] = [0, 1, 0, 0]
        elif c == "C":
            seq_as_list[i] = [0, 0, 1, 0]
        elif c == "G":
            seq_as_list[i] = [0, 0, 0, 1]
        else:
            seq_as_list[i] = [0, 0, 0, 0]

    return np.array(seq_as_list)


class IndexedNucleotides(object):
    """String with various indexes."""

    """Based on LIME official Repo"""

    def __init__(self, raw_string):
        """Initializer.

        Args:
            raw_string: string with raw text in it
        """
        self.raw = raw_string
        self.as_list = list(self.raw)
        self.as_np = np.array(self.as_list)
        self.string_start = np.arange(len(self.raw))
        vocab = {}
        self.inverse_vocab = []
        self.positions = []
        non_vocab = set()
        for i, char in enumerate(self.as_np):
            if char in non_vocab:
                continue
            self.inverse_vocab.append(char)
            self.positions.append(i)
        self.positions = np.array(self.positions)

    def raw_string(self):
        """Returns the original raw string"""
        return self.raw

    def num_nucleotides(self):
        """Returns the number of tokens in the vocabulary for this document."""
        return len(self.inverse_vocab)

    def choose_alt(self, existing):
        nucleotides = ["A", "T", "G", "C"]
        nucleotides.remove(existing)
        return nucleotides[np.random.randint(0, 3)]

    def perturb_nucleotide(self, chars_to_remove):
        mask = np.ones(self.as_np.shape[0], dtype="bool")
        mask[self.__get_idxs(chars_to_remove)] = False
        return "".join(
            [
                self.as_list[i] if mask[i] else self.choose_alt(self.as_list[i])
                for i in range(mask.shape[0])
            ]
        )

    def __get_idxs(self, chars):
        """Returns indexes to appropriate words."""
        return self.positions[chars]
