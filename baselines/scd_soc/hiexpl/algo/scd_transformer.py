from algo.soc_transformer import (
    ExplanationBaseForTransformer,
    SOCForTransformer,
    normalize_logit,
)
from bert.run_classifier import BertTokenizer, predict_and_explain_wrapper_unbatched
from algo.soc_transformer import (
    get_data_iterator_bert,
    bert_id_to_lm_id,
    lm_id_to_bert_id,
    Batch,
)
from bert.modeling import global_state_dict
import torch
import copy
import numpy as np
from utils.args import get_args

args = get_args()


class CDForTransformer(ExplanationBaseForTransformer):
    def __init__(self, model, tree_path, output_path, config, tokenizer):
        super().__init__(model, tree_path, output_path, config, tokenizer)
        self.model = model
        self.tokenizer = tokenizer  # BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir='bert/cache')
        self.tree_path = tree_path
        self.max_seq_length = 128
        self.batch_size = config.batch_size
        self.iterator = get_data_iterator_bert(
            self.tree_path, self.tokenizer, self.max_seq_length, self.batch_size
        )

    def explain_single_transformer(
        self, input_ids, input_mask, segment_ids, region, label=None
    ):
        if self.gpu >= 0:
            input_ids, input_mask, segment_ids = (
                input_ids.to(self.gpu),
                input_mask.to(self.gpu),
                segment_ids.to(self.gpu),
            )
        if not args.task == "tacred":
            score = predict_and_explain_wrapper_unbatched(
                self.model, input_ids, segment_ids, input_mask, region
            )
        else:
            score = predict_and_explain_wrapper_unbatched(
                self.model,
                input_ids,
                segment_ids,
                input_mask,
                region,
                normalizer=normalize_logit,
                label=label,
            )
        return score


class SCDForTransformer(SOCForTransformer):
    def __init__(
        self, target_model, lm_model, vocab, tree_path, output_path, config, tokenizer
    ):
        super().__init__(
            target_model, lm_model, vocab, tree_path, output_path, config, tokenizer
        )
        if args.cd_pad:
            self.sample_num = 1

    def get_states(self, inp, inp_mask, segment_ids, x_regions, nb_regions):
        global_state_dict.init_store_states()

        x_region = x_regions[0]
        nb_region = nb_regions[0]

        inp_length = 0
        for i in range(len(inp_mask)):
            if inp_mask[i] == 1:
                inp_length += 1
            else:
                break

        # mask everything outside the x_region and inside nb region
        inp_lm = copy.copy(inp)
        for i in range(len(inp_lm)):
            if (
                nb_region[0] <= i <= nb_region[1]
                and not x_region[0] <= i <= x_region[1]
            ):
                inp_lm[i] = self.tokenizer.vocab["[PAD]"]

        if not args.task == "tacred":
            inp_th = (
                torch.from_numpy(
                    bert_id_to_lm_id(
                        inp_lm[1 : inp_length - 1], self.tokenizer, self.vocab
                    )
                )
                .long()
                .view(-1, 1)
            )
            inp_length = torch.LongTensor([inp_length - 2])
            fw_pos = torch.LongTensor([min(x_region[1] + 1 - 1, len(inp) - 2)])
            bw_pos = torch.LongTensor([max(x_region[0] - 1 - 1, -1)])

        else:
            inp_th = (
                torch.from_numpy(
                    bert_id_to_lm_id(inp_lm[:inp_length], self.tokenizer, self.vocab)
                )
                .long()
                .view(-1, 1)
            )
            inp_length = torch.LongTensor([inp_length])
            fw_pos = torch.LongTensor([min(x_region[1] + 1, inp_length.item() - 1)])
            bw_pos = torch.LongTensor([max(x_region[0] - 1, 0)])

        if self.gpu >= 0:
            inp_th = inp_th.to(self.gpu)
            inp_length = inp_length.to(self.gpu)
            fw_pos = fw_pos.to(self.gpu)
            bw_pos = bw_pos.to(self.gpu)

        batch = Batch(text=inp_th, length=inp_length, fw_pos=fw_pos, bw_pos=bw_pos)

        inp_enb = []

        max_sample_length = (
            (self.nb_range + 1) if self.nb_method == "ngram" else (inp_th.size(0) + 1)
        )
        fw_sample_outputs, bw_sample_outputs = self.lm_model.sample_n(
            "random",
            batch,
            max_sample_length=max_sample_length,
            sample_num=self.sample_num,
        )

        if not args.cd_pad:
            for sample_i in range(self.sample_num):
                fw_sample_seq, bw_sample_seq = (
                    fw_sample_outputs[:, sample_i].cpu().numpy(),
                    bw_sample_outputs[:, sample_i].cpu().numpy(),
                )
                filled_inp = copy.copy(inp)

                len_bw = x_region[0] - nb_region[0]
                len_fw = nb_region[1] - x_region[1]
                if len_bw > 0:
                    filled_inp[nb_region[0] : x_region[0]] = lm_id_to_bert_id(
                        bw_sample_seq[-len_bw:], self.tokenizer, self.vocab
                    )
                if len_fw > 0:
                    filled_inp[x_region[1] + 1 : nb_region[1] + 1] = lm_id_to_bert_id(
                        fw_sample_seq[:len_fw], self.tokenizer, self.vocab
                    )
                inp_enb.append(filled_inp)

        else:
            filled_inp = copy.copy(inp)
            for i in range(nb_region[0], nb_region[1] + 1):
                if not x_region[0] <= i <= x_region[1]:
                    filled_inp[i] = self.tokenizer.vocab["[PAD]"]
            inp_enb.append(filled_inp)
        inp_enb = np.stack(inp_enb)
        inp_enb = torch.from_numpy(inp_enb).long()
        inp_enb_mask = torch.from_numpy(inp_mask).long()

        if self.gpu >= 0:
            inp_enb = inp_enb.to(self.gpu)
            inp_enb_mask = inp_enb_mask.to(self.gpu)
            segment_ids = segment_ids.to(self.gpu)

        inp_enb_mask = inp_enb_mask.expand(inp_enb.size(0), -1)
        segment_ids = segment_ids.expand(inp_enb.size(0), -1)

        self.model.predict_and_explain(
            inp_enb,
            [[x_region]] * inp_enb.size(0),  ### important
            segment_ids[:, : inp_enb.size(1)],
            inp_enb_mask,
        )

        global_state_dict.init_fetch_states()
        return

    def explain_single_transformer(
        self, input_ids, input_mask, segment_ids, region, label=None
    ):
        inp_flatten = input_ids.view(-1).cpu().numpy()
        inp_mask_flatten = input_mask.view(-1).cpu().numpy()
        if self.nb_method == "ngram":
            mask_regions = self.get_ngram_mask_region(region, inp_flatten)
        else:
            raise NotImplementedError("unknown method %s" % self.nb_method)

        total_len = int(inp_mask_flatten.sum())
        span_len = region[1] - region[0] + 1

        global_state_dict.total_span_len = total_len
        global_state_dict.rel_span_len = span_len

        self.get_states(
            inp_flatten, inp_mask_flatten, segment_ids, [region], mask_regions
        )
        if self.gpu >= 0:
            input_ids, input_mask, segment_ids = (
                input_ids.to(self.gpu),
                input_mask.to(self.gpu),
                segment_ids.to(self.gpu),
            )
        if not args.task == "tacred":
            score = predict_and_explain_wrapper_unbatched(
                self.model, input_ids, segment_ids, input_mask, region
            )
        else:
            score = predict_and_explain_wrapper_unbatched(
                self.model,
                input_ids,
                segment_ids,
                input_mask,
                region,
                normalizer=normalize_logit,
                label=label,
            )
        return score
