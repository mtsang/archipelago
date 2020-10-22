from torch.distributions import Categorical
from nns.layers import *
from utils.args import get_args
from torch.nn import functional as F

args = get_args()


class LSTMLanguageModel(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.hidden_size = config.lm_d_hidden
        self.embed_size = config.lm_d_embed
        self.n_vocab = config.n_embed
        self.gpu = args.gpu

        self.encoder = DynamicEncoder(
            self.n_vocab, self.embed_size, self.hidden_size, self.gpu
        )
        self.fw_proj = nn.Linear(self.hidden_size, self.n_vocab)
        self.bw_proj = nn.Linear(self.hidden_size, self.n_vocab)

        self.loss = nn.CrossEntropyLoss(ignore_index=1)
        self.vocab = vocab

        self.warning_flag = False

    def forward(self, batch):
        inp = batch.text
        inp_len_np = batch.length.cpu().numpy()
        if self.gpu >= 0:
            inp = inp.to(self.gpu)
        output = self.encoder(inp, inp_len_np)
        fw_output, bw_output = (
            output[:, :, : self.hidden_size],
            output[:, :, self.hidden_size :],
        )
        fw_proj, bw_proj = self.fw_proj(fw_output), self.bw_proj(bw_output)

        fw_loss = self.loss(
            fw_proj[:-1].view(-1, fw_proj.size(2)).contiguous(),
            inp[1:].view(-1).contiguous(),
        )
        bw_loss = self.loss(
            bw_proj[1:].view(-1, bw_proj.size(2)).contiguous(),
            inp[:-1].view(-1).contiguous(),
        )
        return fw_loss, bw_loss

    def _sample_n_sequences(
        self, method, direction, token_inp, hidden, length, sample_num
    ):
        outputs = []
        token_inp = token_inp.repeat(1, sample_num)  # [1, N]
        hidden = hidden[0].repeat(1, sample_num, 1), hidden[1].repeat(
            1, sample_num, 1
        )  # [x, N, H]
        for t in range(length):
            output, hidden = self.encoder.rollout(
                token_inp, hidden, direction=direction
            )
            if direction == "fw":
                proj = self.fw_proj(output[:, :, : self.hidden_size])
            elif direction == "bw":
                proj = self.bw_proj(output[:, :, self.hidden_size :])
            proj = proj.squeeze(0)
            if method == "max":
                _, token_inp = torch.max(proj, -1)
                outputs.append(token_inp.view(-1))
            elif method == "random":
                dist = Categorical(F.softmax(proj, -1))
                token_inp = dist.sample()
                outputs.append(token_inp)
            token_inp = token_inp.view(1, -1)
        if direction == "bw":
            outputs = list(reversed(outputs))
        outputs = torch.stack(outputs)
        return outputs

    def sample_n(self, method, batch, max_sample_length, sample_num):
        inp = batch.text
        inp_len_np = batch.length.cpu().numpy()
        batch_size = inp.size(1)
        assert batch_size == 1

        pad_inp1 = torch.LongTensor([self.vocab.stoi["<s>"]] * inp.size(1)).view(1, -1)
        pad_inp2 = torch.LongTensor([self.vocab.stoi["</s>"]] * inp.size(1)).view(1, -1)

        if self.gpu >= 0:
            inp = inp.to(self.gpu)
            pad_inp1 = pad_inp1.to(self.gpu)
            pad_inp2 = pad_inp2.to(self.gpu)

        padded_inp = torch.cat([pad_inp1, inp, pad_inp2], 0)
        assert padded_inp.max().item() < self.n_vocab
        assert inp_len_np[0] + 2 <= padded_inp.size(0)
        padded_enc_out, (padded_hidden_states, padded_cell_states) = self.encoder(
            padded_inp, inp_len_np + 2, return_all_states=True
        )  # [T+2,B,H]

        # extract forward hidden state
        assert 0 <= batch.fw_pos.item() - 1 <= padded_enc_out.size(0) - 1
        assert 0 <= batch.fw_pos.item() <= padded_enc_out.size(0) - 1

        fw_hidden_state = padded_hidden_states.index_select(0, batch.fw_pos - 1)[0]
        fw_cell_state = padded_cell_states.index_select(0, batch.fw_pos - 1)[0]
        fw_next_token = padded_inp.index_select(0, batch.fw_pos).view(1, -1)

        # extract backward hidden state
        assert 0 <= batch.bw_pos.item() + 3 <= padded_enc_out.size(0) - 1
        assert 0 <= batch.bw_pos.item() + 2 <= padded_enc_out.size(0) - 1
        # batch
        bw_hidden_state = padded_hidden_states.index_select(0, batch.bw_pos + 3)[0]
        bw_cell_state = padded_cell_states.index_select(0, batch.bw_pos + 3)[0]
        # torch.cat([bw_hidden[:,:,:self.hidden_size], bw_hidden[:,:,self.hidden_size:]], 0)
        bw_next_token = padded_inp.index_select(0, batch.bw_pos + 2).view(1, -1)

        fw_sample_outputs = self._sample_n_sequences(
            method,
            "fw",
            fw_next_token,
            (fw_hidden_state, fw_cell_state),
            max_sample_length,
            sample_num,
        )
        bw_sample_outputs = self._sample_n_sequences(
            method,
            "bw",
            bw_next_token,
            (bw_hidden_state, bw_cell_state),
            max_sample_length,
            sample_num,
        )

        self.filter_special_tokens(fw_sample_outputs)
        self.filter_special_tokens(bw_sample_outputs)

        return fw_sample_outputs, bw_sample_outputs

    def filter_special_tokens(self, m):
        for i in range(m.size(0)):
            for j in range(m.size(1)):
                if m[i, j] >= self.n_vocab - 2:
                    m[i, j] = 0
