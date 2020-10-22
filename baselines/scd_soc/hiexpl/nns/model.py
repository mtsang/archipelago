import torch
import torch.nn as nn
import numpy as np
from torch.nn import init


class LSTMSentiment(nn.Module):
    def __init__(self, config, match_length=False):
        super(LSTMSentiment, self).__init__()
        self.hidden_dim = config.d_hidden
        self.vocab_size = config.n_embed
        self.emb_dim = config.d_embed
        self.batch_size = config.batch_size
        self.use_gpu = config.gpu >= 0
        self.num_labels = config.d_out
        self.embed = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=1)
        self.match_length = match_length

        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_dim)
        self.hidden_to_label = nn.Linear(self.hidden_dim, self.num_labels)

    def forward(self, batch):
        if hasattr(batch, "vec"):
            vecs = batch.vec
        else:
            if self.use_gpu:
                inp = batch.text.long().cuda()
            else:
                inp = batch.text.long()
            vecs = self.embed(inp)
        lstm_out, hidden = self.lstm(vecs)  # [T, B, H]

        if hasattr(self, "match_length") and self.match_length:
            if hasattr(batch, "length"):
                length = batch.length.cpu().numpy()
            elif vecs.size(1) == 1:  # batch size is 1
                length = np.array([vecs.size(0)])
            else:
                length = np.array([vecs.size(0)] * vecs.size(1))
                print("Warning: length is missing")
            hidden_state = []
            for i in range(length.shape[0]):
                hidden_state.append(lstm_out[length[i] - 1, i])
            hidden_state = torch.stack(hidden_state)  # [B,H]
        else:
            hidden_state = lstm_out[-1]
        logits = self.hidden_to_label(hidden_state)
        return logits


class LSTMMeanSentiment(LSTMSentiment):
    def __init__(self, config, match_length=False):
        super().__init__(config, match_length)

    def forward(self, batch):
        if hasattr(batch, "vec"):
            vecs = batch.vec
        else:
            if self.use_gpu:
                inp = batch.text.long().cuda()
            else:
                inp = batch.text.long()
            vecs = self.embed(inp)
        lstm_out, hidden = self.lstm(vecs)  # [T, B, H]

        if self.match_length:
            if hasattr(batch, "length"):
                length = batch.length.cpu().numpy()
            elif vecs.size(1) == 1:  # batch size is 1
                length = np.array([vecs.size(0)])
            else:
                length = np.array([vecs.size(0)] * vecs.size(1))
                # print('Warning: length is missing')
            hidden_state = []
            for i in range(length.shape[0]):
                hidden_state.append(torch.mean(lstm_out[: length[i], i], 0))
            hidden_state = torch.stack(hidden_state)  # [B,H]
        else:
            hidden_state = torch.mean(lstm_out, 0)
        logits = self.hidden_to_label(hidden_state)
        return logits


class LSTMMeanRE(LSTMSentiment):
    def __init__(self, config, match_length=False):
        super().__init__(config, match_length)
        self.drop = nn.Dropout(0.5)
        self.pos_embed = nn.Embedding(
            config.pos_size, config.offset_emb_dim, padding_idx=1
        )
        self.ner_embed = nn.Embedding(
            config.ner_size, config.offset_emb_dim, padding_idx=1
        )
        self.lstm = LSTMLayer(
            self.emb_dim + config.offset_emb_dim * 2, self.hidden_dim, 1, 0.5, True
        )
        self.hidden_to_label = nn.Linear(self.hidden_dim, config.d_out)

    def forward(self, batch):
        if hasattr(batch, "vec"):
            vecs = batch.vec
        else:
            inp_tokens = batch.text.long().cuda()
            inp_pos = batch.pos.long().cuda()
            inp_ner = batch.ner.long().cuda()

            token_vecs = self.embed(inp_tokens)
            pos_vecs = self.pos_embed(inp_pos)
            ner_vecs = self.ner_embed(inp_ner)
            vecs = torch.cat([token_vecs, pos_vecs, ner_vecs], -1)  # [T, B, H]
        vecs = self.drop(vecs)
        h0, c0 = self.zero_state(vecs.size(1))
        if not hasattr(batch, "length"):
            batch.length = torch.LongTensor([vecs.size(0)] * vecs.size(1)).cuda()
        lstm_out, (ht, ct) = self.lstm(vecs, batch.length, (h0, c0))

        hidden = self.drop(ht[-1, :, :])
        logits = self.hidden_to_label(hidden)
        return logits

    def init_weights(self):
        self.pos_embed.weight.data[2:, :].uniform_(-1.0, 1.0)
        self.ner_embed.weight.data[2:, :].uniform_(-1.0, 1.0)

        self.hidden_to_label.bias.data.fill_(0)
        init.xavier_uniform_(
            self.hidden_to_label.weight, gain=1
        )  # initialize linear layer

    def zero_state(self, batch_size):
        state_shape = (1, batch_size, self.hidden_dim)
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        return h0.cuda(), c0.cuda()


class LSTMLayer(nn.Module):
    # Copyright 2017 The Board of Trustees of The Leland Stanford Junior University
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    """A wrapper for LSTM with sequence packing."""

    def __init__(self, emb_dim, hidden_dim, num_layers, dropout, use_cuda):
        super(LSTMLayer, self).__init__()
        self.rnn = nn.LSTM(
            emb_dim, hidden_dim, num_layers, batch_first=False, dropout=dropout
        )
        self.use_cuda = use_cuda

    def forward(self, x, x_lens, init_state):
        """
        x: batch_size * feature_size * seq_len
        x_mask : batch_size * seq_len
        """
        x_lens = x_lens.cuda()
        _, idx_sort = torch.sort(x_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lens = list(x_lens[idx_sort])

        # sort by seq lens
        x = x.index_select(1, idx_sort)
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=False)
        rnn_output, (ht, ct) = self.rnn(rnn_input, init_state)
        rnn_output = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=False)[0]

        # unsort
        rnn_output = rnn_output.index_select(1, idx_unsort)
        ht = ht.index_select(1, idx_unsort)
        ct = ct.index_select(1, idx_unsort)
        return rnn_output, (ht, ct)
