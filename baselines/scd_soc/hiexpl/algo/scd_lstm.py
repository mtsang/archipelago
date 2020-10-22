import torch
from algo.cd_func import CD
from algo.scd_func import CD_gpu, get_lstm_states
from algo.soc_lstm import (
    ExplanationBase,
    SOCForLSTM,
    Batch,
    append_extra_input,
    normalize_logit,
)
import copy
from utils.args import get_args

args = get_args()


class CDForLSTM(ExplanationBase):
    def __init__(
        self,
        target_model,
        data_iterator,
        vocab,
        pad_variation,
        tree_path,
        output_path,
        config,
        pad_idx=1,
    ):
        super().__init__(
            target_model, data_iterator, vocab, tree_path, output_path, config, pad_idx
        )
        self.pad_variation = pad_variation

    def explain_single(self, inp, inp_id, region, extra_input=None):
        """
        :param region: the input region to be explained
        :param inp: numpy array
        :param inp_id: int
        :return:
        """
        inp = torch.from_numpy(inp).long().view(-1, 1)
        if self.gpu >= 0:
            inp = inp.to(self.gpu)
        batch = Batch(text=inp)
        if extra_input is not None:
            append_extra_input(batch, extra_input)
        rel_scores, irrel_score = CD(batch, self.model, [region])
        if rel_scores.shape[0] == 2:
            score = rel_scores[0] - rel_scores[1]
        else:
            gt_label = extra_input["gt_label"]
            contrib_logits = torch.from_numpy(rel_scores).cuda().view(1, -1)
            score = normalize_logit(contrib_logits, gt_label)
            score = score.item()
        return score


class SCDForLSTM(SOCForLSTM):
    def __init__(
        self,
        target_model,
        lm_model,
        data_iterator,
        vocab,
        tree_path,
        output_path,
        config,
        pad_idx=1,
    ):
        super().__init__(
            target_model,
            lm_model,
            data_iterator,
            vocab,
            tree_path,
            output_path,
            config,
            pad_idx,
        )
        self.sample_num = config.sample_n if not args.cd_pad else 1

    def get_states(self, inp, x_regions, nb_regions, extra_input):
        # suppose only have one x_region and one nb_region
        x_region = x_regions[0]
        nb_region = nb_regions[0]

        inp_length = torch.LongTensor([len(inp)])
        fw_pos = torch.LongTensor([min(x_region[1] + 1, len(inp))])
        bw_pos = torch.LongTensor([max(x_region[0] - 1, -1)])

        inp_lm = copy.copy(inp)
        for i in range(len(inp_lm)):
            if (
                nb_region[0] <= i <= nb_region[1]
                and not x_region[0] <= i <= x_region[1]
            ):
                inp_lm[i] = 1
        inp_th = torch.from_numpy(inp_lm).long().view(-1, 1)

        if self.gpu >= 0:
            inp_th = inp_th.to(self.gpu)
            inp_length = inp_length.to(self.gpu)
            fw_pos = fw_pos.to(self.gpu)
            bw_pos = bw_pos.to(self.gpu)

        batch = Batch(text=inp_th, length=inp_length, fw_pos=fw_pos, bw_pos=bw_pos)

        all_filled_inp = []
        max_sample_length = (
            (self.nb_range + 1) if self.nb_method == "ngram" else (inp_th.size(0) + 1)
        )

        if not args.cd_pad:
            fw_sample_outputs, bw_sample_outputs = self.lm_model.sample_n(
                "random",
                batch,
                max_sample_length=max_sample_length,
                sample_num=self.sample_num,
            )
            for sample_i in range(self.sample_num):
                fw_sample_seq, bw_sample_seq = (
                    fw_sample_outputs[:, sample_i].cpu().numpy(),
                    bw_sample_outputs[:, sample_i].cpu().numpy(),
                )
                filled_inp = copy.copy(inp)
                len_bw = x_region[0] - nb_region[0]
                len_fw = nb_region[1] - x_region[1]
                if len_bw > 0:
                    filled_inp[nb_region[0] : x_region[0]] = bw_sample_seq[-len_bw:]
                if len_fw > 0:
                    filled_inp[x_region[1] + 1 : nb_region[1] + 1] = fw_sample_seq[
                        :len_fw
                    ]

                filled_inp = torch.from_numpy(filled_inp).long()
                if self.gpu >= 0:
                    filled_inp = filled_inp.to(self.gpu)
                all_filled_inp.append(filled_inp)
        else:
            # pad the nb region to 1
            filled_inp = copy.copy(inp)
            for i in range(nb_region[0], nb_region[1] + 1):
                if not x_region[0] <= i <= x_region[1]:
                    filled_inp[i] = 1
            filled_inp = torch.from_numpy(filled_inp).long()
            if self.gpu >= 0:
                filled_inp = filled_inp.to(self.gpu)
            all_filled_inp.append(filled_inp)

        all_filled_inp = torch.stack(all_filled_inp, -1)  # [T,B]
        batch = Batch(text=all_filled_inp)

        if extra_input is not None:
            append_extra_input(batch, extra_input)
        all_states = get_lstm_states(batch, self.model, self.gpu)

        return all_states

    def explain_single(self, inp, inp_id, region, extra_input=None):
        """
        :param region: the input region to be explained
        :param inp: numpy array
        :param inp_id: int
        :return:
        """
        if self.nb_method == "tree":
            tree = self.trees[inp_id]
            mask_regions = self.get_tree_mask_region(tree, region, inp)
        elif self.nb_method == "ngram":
            mask_regions = self.get_ngram_mask_region(region, inp)
        else:
            raise NotImplementedError("unknown method %s" % self.nb_method)
        with torch.no_grad():
            if self.sample_num > 0:
                states = self.get_states(inp, [region], mask_regions, extra_input)
            else:
                states = None
            inp = torch.from_numpy(inp).long().view(-1, 1)
            if self.gpu >= 0:
                inp = inp.to(self.gpu)
            batch = Batch(text=inp)
            if extra_input is not None:
                append_extra_input(batch, extra_input)
            rel_scores, irrel_scores, _ = CD_gpu(
                batch, self.model, [region], states, gpu=self.gpu
            )
        if rel_scores.shape[0] == 2:
            score = rel_scores[0] - rel_scores[1]
        else:
            gt_label = extra_input["gt_label"]
            contrib_logits = rel_scores.view(1, -1)
            score = normalize_logit(contrib_logits, gt_label)
            score = score.item()
        return score
