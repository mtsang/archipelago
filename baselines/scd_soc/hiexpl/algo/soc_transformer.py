from algo.soc_lstm import *
from bert.run_classifier import (
    BertTokenizer,
    TensorDataset,
    SequentialSampler,
    DataLoader,
)
from bert.run_lm_finetuning import BertForPreTraining
import torch
from utils.parser import get_span_to_node_mapping
from utils.reader import get_examples_sst, get_examples_yelp, get_examples_tacred
import numpy as np
import copy
import pickle
from utils.args import get_args

DotDict = Batch
args = get_args()


def convert_examples_to_features_sst(examples, max_seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = example.text
        mapping = example.mapping
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[: (max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        mapping += [-1] * (max_seq_length - len(mapping))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = example.label

        features.append(
            DotDict(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                offset=example.offset,
                mapping=mapping,
            )
        )
    return features


def bert_id_to_lm_id(arr, bert_tokenizer, lm_vocab):
    tmp_tokens = bert_tokenizer.convert_ids_to_tokens(arr)
    tokens = []
    conv_dict = {"[UNK]": "<unk>", "[PAD]": "<pad>"}
    for w in tmp_tokens:
        tokens.append(conv_dict.get(w, w))
    lm_ids = [lm_vocab.stoi.get(token, 0) for token in tokens]
    return np.array(lm_ids, dtype=np.int32)


def lm_id_to_bert_id(arr, bert_tokenizer, lm_vocab):
    #     print(arr, arr.shape, len(lm_vocab.itos))
    tmp_tokens = []
    for x in arr.tolist():
        if x >= len(lm_vocab.itos):
            tmp_token = "UNKUWN"
        else:
            tmp_token = lm_vocab.itos[x]
        tmp_tokens.append(tmp_token)
    #     tmp_tokens = [lm_vocab.itos[x] for x in arr.tolist()]
    tokens = []
    conv_dict = {"<unk>": "[UNK]", "<pad>": "[PAD]"}
    for w in tmp_tokens:
        tokens.append(conv_dict.get(w, w))
    bert_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
    return np.array(bert_ids, dtype=np.int32)


def get_data_iterator_bert(
    tree_path, tokenizer, max_seq_length, batch_size, label_vocab=None
):
    if args.task == "sst":
        eval_examples = get_examples_sst(
            tree_path, train_lm=False, bert_tokenizer=tokenizer
        )
    elif args.task == "yelp":
        eval_examples = get_examples_yelp(tree_path, False, tokenizer)
    elif args.task == "tacred":
        eval_examples = get_examples_tacred(tree_path, False, tokenizer)
        label_vocab = pickle.load(open("vocab/vocab_tacred_bert.pkl.relation", "rb"))
    else:
        raise ValueError
    eval_features = convert_examples_to_features_sst(
        eval_examples, max_seq_length, tokenizer
    )
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features], dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features], dtype=torch.long
    )
    if args.task != "tacred":
        all_label_ids = torch.tensor(
            [f.label_id for f in eval_features], dtype=torch.long
        )
    else:
        all_label_ids = torch.tensor(
            [label_vocab.stoi[f.label_id] for f in eval_features], dtype=torch.long
        )
    all_offsets = torch.tensor([f.offset for f in eval_features], dtype=torch.long)
    all_mappings = torch.tensor([f.mapping for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(
        all_input_ids,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
        all_offsets,
        all_mappings,
    )
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
    return eval_dataloader


class ExplanationBaseForTransformer(ExplanationBase):
    def __init__(self, model, tree_path, output_path, config, tokenizer):
        super().__init__(model, None, None, tree_path, output_path, config)
        self.model = model
        self.tokenizer = tokenizer  # BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir='bert/cache')
        self.tree_path = tree_path
        self.max_seq_length = 128
        self.batch_size = config.batch_size
        self.iterator = self.get_data_iterator()

    def get_data_iterator(self):
        return get_data_iterator_bert(
            self.tree_path, self.tokenizer, self.max_seq_length, self.batch_size
        )

    def occlude_input_with_masks(self, inp, inp_mask, x_regions, nb_regions):
        region_indicator = np.zeros(len(inp), dtype=np.int32)
        for region in nb_regions:
            region_indicator[region[0] : region[1] + 1] = 2
        for region in x_regions:
            region_indicator[region[0] : region[1] + 1] = 1
        # input expectation over neighbourhood
        inp_enb = copy.copy(inp)
        inp_mask_enb = copy.copy(inp_mask)
        # input expectation over neighbourhood and selected span
        inp_ex = copy.copy(inp)
        inp_mask_ex = copy.copy(inp_mask)

        inp_enb, inp_mask_enb = self.mask_region_masked(
            inp_enb, inp_mask_enb, region_indicator, [2]
        )
        inp_ex, inp_mask_ex = self.mask_region_masked(
            inp_ex, inp_mask_ex, region_indicator, [1, 2]
        )

        return inp_enb, inp_mask_enb, inp_ex, inp_mask_ex

    def mask_region_masked(self, inp, inp_mask, region_indicator, mask_value):
        new_seq = []
        new_mask_seq = []
        for i in range(len(region_indicator)):
            if region_indicator[i] not in mask_value:
                new_seq.append(inp[i])
            else:
                new_seq.append(self.tokenizer.vocab["[PAD]"])
            new_mask_seq.append(inp_mask[i])
        if not new_seq:
            new_seq.append(self.tokenizer.vocab["[PAD]"])
        new_seq = np.array(new_seq)
        new_mask_seq = np.array(new_mask_seq)
        return new_seq, new_mask_seq

    def get_ngram_mask_region(self, region, inp):
        # find the [PAD] token
        idx = 0
        while idx < len(inp) and inp[idx] != 0:
            idx += 1
        if not self.nb_unidirectional:
            return [
                (
                    max(region[0] - self.nb_range, 1),
                    min(region[1] + self.nb_range, idx - 2),
                )
            ]
        else:
            return (
                [(max(1, region[0]), min(region[1] + self.nb_range, idx - 2))]
                if not args.task == "tacred"
                else [(max(0, region[0]), min(region[1] + self.nb_range, idx - 1))]
            )

    def explain_single_transformer(
        self, input_ids, input_mask, segment_ids, region, label=None
    ):
        inp_flatten = input_ids.view(-1).cpu().numpy()
        inp_mask_flatten = input_mask.view(-1).cpu().numpy()
        mask_regions = self.get_ngram_mask_region(region, inp_flatten)

        inp_enb, inp_mask_enb, inp_ex, inp_mask_ex = self.occlude_input_with_masks(
            inp_flatten, inp_mask_flatten, [region], mask_regions
        )

        inp_enb, inp_mask_enb, inp_ex, inp_mask_ex = (
            torch.from_numpy(inp_enb).long().view(1, -1),
            torch.from_numpy(inp_mask_enb).long().view(1, -1),
            torch.from_numpy(inp_ex).long().view(1, -1),
            torch.from_numpy(inp_mask_ex).long().view(1, -1),
        )

        if self.gpu >= 0:
            inp_enb, inp_mask_enb, inp_ex, inp_mask_ex = (
                inp_enb.to(self.gpu),
                inp_mask_enb.to(self.gpu),
                inp_ex.to(self.gpu),
                inp_mask_ex.to(self.gpu),
            )
            segment_ids = segment_ids.to(self.gpu)
        logits_enb = self.model(
            inp_enb, segment_ids[:, : inp_enb.size(1)], inp_mask_enb
        )
        logits_ex = self.model(inp_ex, segment_ids[:, : inp_ex.size(1)], inp_mask_ex)
        contrib_logits = logits_enb - logits_ex  # [1 * C]
        if contrib_logits.size(1) == 2:
            contrib_score = contrib_logits[0, 1] - contrib_logits[0, 0]
        else:
            contrib_score = normalize_logit(contrib_logits, label)
        return contrib_score.item()

    def agglomerate(
        self,
        inputs,
        percentile_include,
        method,
        sweep_dim,
        dataset,
        num_iters=5,
        subtract=True,
        absolute=True,
        label=None,
    ):
        text_orig = inputs[0].cpu().clone().numpy().transpose((1, 0))
        for t in range(text_orig.shape[0]):
            if text_orig[t, 0] == 0:
                text_orig = text_orig[:t]
                break
        text_len = text_orig.shape[0]
        score_orig = self.explain_single_transformer(
            *inputs, region=[1, text_len - 2], label=label
        )
        # get scores
        texts = gen_tiles(text_orig, method=method, sweep_dim=sweep_dim)
        texts = texts.transpose()

        starts, stops = tiles_to_cd(texts)

        scores = []
        for (start, stop) in zip(starts, stops):
            score = self.explain_single_transformer(
                *inputs, region=[start, stop], label=label
            )
            scores.append(score)

        # threshold scores
        mask = threshold_scores(scores, percentile_include, absolute=absolute)

        # initialize lists
        scores_list = [np.copy(scores)]
        mask_list = [mask]
        comps_list = []
        comp_scores_list = [{0: score_orig}]

        # iterate
        for step in range(num_iters):
            # find connected components for regions
            comps = np.copy(measure.label(mask_list[-1], background=0, connectivity=1))

            # loop over components
            comp_scores_dict = {}
            for comp_num in range(1, np.max(comps) + 1):

                # make component tile
                comp_tile_bool = comps == comp_num
                comp_tile = gen_tile_from_comp(text_orig, comp_tile_bool, method)

                # make tiles around component
                border_tiles = gen_tiles_around_baseline(
                    text_orig, comp_tile_bool, method=method, sweep_dim=sweep_dim
                )

                # predict for all tiles
                # format tiles into batch
                tiles_concat = np.hstack(
                    (comp_tile, np.squeeze(border_tiles[0]).transpose())
                )

                starts, stops = tiles_to_cd(tiles_concat)
                scores_all = []
                for (start, stop) in zip(starts, stops):
                    score = self.explain_single_transformer(
                        *inputs, region=[start, stop], label=label
                    )
                    scores_all.append(score)

                score_comp = np.copy(scores_all[0])
                scores_border_tiles = np.copy(scores_all[1:])

                # store the predicted class scores
                comp_scores_dict[comp_num] = np.copy(score_comp)

                # update pixel scores
                tiles_idxs = border_tiles[1]
                for i, idx in enumerate(tiles_idxs):
                    scores[idx] = scores_border_tiles[i] - score_comp

            # get class preds and thresholded image
            scores = np.array(scores)
            scores[mask_list[-1]] = np.nan
            mask = threshold_scores(scores, percentile_include, absolute=absolute)

            # add to lists
            scores_list.append(np.copy(scores))
            mask_list.append(mask_list[-1] + mask)
            comps_list.append(comps)
            comp_scores_list.append(comp_scores_dict)

            if np.sum(mask) == 0:
                break

        # pad first image
        comps_list = [np.zeros(text_orig.size, dtype=np.int)] + comps_list

        return {
            "scores_list": scores_list,  # arrs of scores (nan for selected)
            "mask_list": mask_list,  # boolean arrs of selected
            "comps_list": comps_list,  # arrs of comps with diff number for each comp
            "comp_scores_list": comp_scores_list,  # dicts with score for each comp
            "score_orig": score_orig,
        }  # original score

    def map_lm_to_bert_token(self, lm_idx, mapping):
        left = 0 if lm_idx == 0 else mapping[lm_idx - 1]
        right = mapping[lm_idx] - 1
        return left, right

    def explain_sst(self):
        f = open(self.output_path, "w")
        all_contribs = []
        cnt = 0
        for batch_idx, (
            input_ids,
            input_mask,
            segment_ids,
            label_ids,
            offsets,
            mappings,
        ) in enumerate(self.iterator):
            if batch_idx < self.batch_start:
                continue

            inp = input_ids.view(-1).cpu().numpy()
            inp_id = offsets.item()
            mappings = mappings.view(-1).cpu().numpy()
            span2node, node2span = get_span_to_node_mapping(self.trees[inp_id])
            spans = list(span2node.keys())
            repr_spans = []
            contribs = []
            for span in spans:
                if type(span) is int:
                    span = (span, span)
                bert_span = (
                    self.map_lm_to_bert_token(span[0], mappings)[0],
                    self.map_lm_to_bert_token(span[1], mappings)[1],
                )
                # add 1 to spans since transformer inputs has [CLS]
                repr_spans.append(bert_span)
                bert_span = (bert_span[0] + 1, bert_span[1] + 1)
                contrib = self.explain_single_transformer(
                    input_ids, input_mask, segment_ids, bert_span
                )
                contribs.append(contrib)
            all_contribs.append(contribs)

            s = self.repr_result_region(inp, repr_spans, contribs)
            f.write(s + "\n")
            f.flush()

            print("finished %d" % batch_idx)
            cnt += 1
            if batch_idx == self.batch_stop - 1:
                break
        f.close()
        return all_contribs

    def explain_agg(self, dataset):
        f = open(self.output_path, "wb")
        all_tabs = []

        for batch_idx, (
            input_ids,
            input_mask,
            segment_ids,
            label_ids,
            offsets,
            mappings,
        ) in enumerate(self.iterator):
            if batch_idx < self.batch_start:
                continue
            # get prediction
            if label_ids.item() == 0:
                continue
            logits = self.model(input_ids.cuda(), input_mask.cuda(), segment_ids.cuda())
            _, pred = logits.max(-1)
            if pred.item() != label_ids.item():
                continue

            inp = input_ids.view(-1).cpu().numpy()

            lists = self.agglomerate(
                (input_ids, input_mask, segment_ids),
                percentile_include=90,
                method="cd",
                sweep_dim=1,
                dataset=dataset,
                num_iters=10,
                label=label_ids.item(),
            )
            lists = collapse_tree(lists)
            seq_len = lists["scores_list"][0].shape[0]
            data = lists_to_tabs(lists, seq_len)
            text = " ".join(self.tokenizer.convert_ids_to_tokens(inp)[:seq_len])
            label_name = self.label_vocab.itos[label_ids.item()]
            all_tabs.append({"tab": data, "text": text, "label": label_name})
            print("finished %d" % batch_idx)

            if batch_idx >= self.batch_stop - 1:
                break
        pickle.dump(all_tabs, f)
        f.close()
        return all_tabs

    def explain_token(self, dataset):
        f = open(self.output_path, "w")
        all_contribs = []
        cnt = 0
        for batch_idx, (
            input_ids,
            input_mask,
            segment_ids,
            label_ids,
            offsets,
            mappings,
        ) in enumerate(self.iterator):
            if batch_idx < self.batch_start:
                continue
            if args.task == "tacred":
                if label_ids.item() == 0:
                    continue
                logits = self.model(
                    input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()
                )
                _, pred = logits.max(-1)
                if pred.item() != label_ids.item():
                    continue

            inp = input_ids.view(-1).cpu().numpy()
            inp_id = offsets.item()
            mappings = mappings.view(-1).cpu().numpy()

            if 0 in inp.tolist():
                length = inp.tolist().index(0)  # [PAD]
            else:
                length = len(inp)
            repr_spans = []
            contribs = []
            for span in range(length - 2):
                if type(span) is int:
                    span = (span, span)
                # bert_span = self.map_lm_to_bert_token(span[0], mappings)[0], self.map_lm_to_bert_token(span[1], mappings)[1]
                bert_span = span
                # add 1 to spans since transformer inputs has [CLS]
                repr_spans.append(bert_span)
                # if not args.task == 'tacred':
                bert_span = (bert_span[0] + 1, bert_span[1] + 1)
                contrib = self.explain_single_transformer(
                    input_ids, input_mask, segment_ids, bert_span, label_ids.item()
                )
                contribs.append(contrib)
            all_contribs.append(contribs)

            s = self.repr_result_region(
                inp, repr_spans, contribs, label=label_ids.item()
            )
            f.write(s + "\n")

            print("finished %d" % batch_idx)
            cnt += 1
            if batch_idx == self.batch_stop - 1:
                break
        f.close()
        return all_contribs

    def map_lm_to_bert_span(self, span, mappings):
        span = (
            self.map_lm_to_bert_token(span[0], mappings)[0],
            self.map_lm_to_bert_token(span[1], mappings)[1],
        )
        # add 1 to spans since transformer inputs has [CLS]
        span = (span[0] + 1, span[1] + 1)
        return span

    def repr_result_region(self, inp, spans, contribs, label=None):
        tokens = self.tokenizer.convert_ids_to_tokens(inp)
        outputs = []
        assert len(spans) == len(contribs)
        for span, contrib in zip(spans, contribs):
            outputs.append((" ".join(tokens[span[0] + 1 : span[1] + 2]), contrib))
        output_str = " ".join(["%s %.6f\t" % (x, y) for x, y in outputs])
        if (
            label is not None
            and hasattr(self, "label_vocab")
            and self.label_vocab is not None
        ):
            output_str = self.label_vocab.itos[label] + "\t" + output_str
        return output_str

    def demo(self):
        f = open(self.output_path, "w")
        while True:
            l = input("sentence?")
            inp_word = ["[CLS]"] + l.strip().split() + ["[SEP]"]
            inp_word_id = self.tokenizer.convert_tokens_to_ids(inp_word)
            inp = (
                torch.from_numpy(np.array(inp_word_id, dtype=np.int32))
                .long()
                .view(1, -1)
            )
            input_mask = torch.ones_like(inp).long().view(1, -1)
            segment_ids = torch.zeros_like(inp).view(1, -1).to(self.gpu).long()
            spans = [(x, x) for x in range(0, len(inp_word_id) - 2)]

            contribs = []
            for span in spans:
                span = (span[0] + 1, span[1] + 1)
                contrib = self.explain_single_transformer(
                    inp, input_mask, segment_ids, span
                )
                contribs.append(contrib)

            s = self.repr_result_region(inp.view(-1).cpu().numpy(), spans, contribs)
            f.write(s + "\n")
            print(s)


class SOCForTransformer(ExplanationBaseForTransformer):
    def __init__(
        self, target_model, lm_model, vocab, tree_path, output_path, config, tokenizer
    ):
        super().__init__(target_model, tree_path, output_path, config, tokenizer)
        self.model = target_model
        self.lm_model = lm_model
        self.tokenizer = tokenizer  # BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir='bert/cache')
        self.tree_path = tree_path
        self.max_seq_length = 128
        self.batch_size = config.batch_size
        self.sample_num = config.sample_n
        self.vocab = vocab
        self.iterator = self.get_data_iterator()

        self.use_bert_lm = config.use_bert_lm
        self.bert_lm = None
        if self.use_bert_lm:
            self.bert_lm = BertForPreTraining.from_pretrained(
                "bert-base-uncased", cache_dir="./bert/cache"
            ).to(self.gpu)

        self.feasible_bert_ids = self.get_feasible_bert_ids()

    def get_feasible_bert_ids(self):
        s = set()
        for w in self.vocab.stoi:
            s.add(self.tokenizer.vocab.get(w, -1))
        return s

    def get_data_iterator(self):
        return get_data_iterator_bert(
            self.tree_path, self.tokenizer, self.max_seq_length, self.batch_size
        )

    def score(self, inp_bak, inp_mask, segment_ids, x_regions, nb_regions, label):
        x_region = x_regions[0]
        nb_region = nb_regions[0]
        inp = copy.copy(inp_bak)

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

        inp_th = (
            torch.from_numpy(
                bert_id_to_lm_id(inp_lm[1 : inp_length - 1], self.tokenizer, self.vocab)
            )
            .long()
            .view(-1, 1)
        )
        inp_length = torch.LongTensor([inp_length - 2])
        fw_pos = torch.LongTensor([min(x_region[1] + 1 - 1, len(inp) - 2)])
        bw_pos = torch.LongTensor([max(x_region[0] - 1 - 1, -1)])

        if self.gpu >= 0:
            inp_th = inp_th.to(self.gpu)
            inp_length = inp_length.to(self.gpu)
            fw_pos = fw_pos.to(self.gpu)
            bw_pos = bw_pos.to(self.gpu)

        batch = Batch(text=inp_th, length=inp_length, fw_pos=fw_pos, bw_pos=bw_pos)

        inp_enb, inp_ex = [], []
        inp_ex_mask = []

        max_sample_length = (
            (self.nb_range + 1) if self.nb_method == "ngram" else (inp_th.size(0) + 1)
        )
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
                filled_inp[nb_region[0] : x_region[0]] = lm_id_to_bert_id(
                    bw_sample_seq[-len_bw:], self.tokenizer, self.vocab
                )
            if len_fw > 0:
                filled_inp[x_region[1] + 1 : nb_region[1] + 1] = lm_id_to_bert_id(
                    fw_sample_seq[:len_fw], self.tokenizer, self.vocab
                )
            inp_enb.append(filled_inp)

            filled_ex, mask_ex = [], []
            for i in range(len(filled_inp)):
                if not x_region[0] <= i <= x_region[1]:
                    filled_ex.append(filled_inp[i])
                    mask_ex.append(inp_mask[i])
                else:
                    filled_ex.append(self.tokenizer.vocab["[PAD]"])
                    mask_ex.append(inp_mask[i])
            filled_ex = np.array(filled_ex, dtype=np.int32)
            mask_ex = np.array(mask_ex, dtype=np.int32)
            inp_ex.append(filled_ex)
            inp_ex_mask.append(mask_ex)

        inp_enb, inp_ex = np.stack(inp_enb), np.stack(inp_ex)
        inp_ex_mask = np.stack(inp_ex_mask)
        inp_enb, inp_ex = (
            torch.from_numpy(inp_enb).long(),
            torch.from_numpy(inp_ex).long(),
        )
        inp_enb_mask, inp_ex_mask = (
            torch.from_numpy(inp_mask).long(),
            torch.from_numpy(inp_ex_mask).long(),
        )

        #         if self.gpu >= 0:
        #             inp_enb, inp_ex = inp_enb.to(self.gpu), inp_ex.to(self.gpu)
        #             inp_enb_mask, inp_ex_mask = inp_enb_mask.to(self.gpu), inp_ex_mask.to(self.gpu)
        #             segment_ids = segment_ids.to(self.gpu)

        inp_enb_mask = inp_enb_mask.expand(inp_enb.size(0), -1)
        segment_ids = segment_ids.expand(inp_enb.size(0), -1)

        batch_size = 20
        num_batches = int(np.ceil(inp_enb.shape[0] / batch_size))

        logits_enb = []
        logits_ex = []

        for i in range(num_batches):
            batch_enb = inp_enb[i * batch_size : (i + 1) * batch_size].to(self.gpu)
            batch_ex = inp_ex[i * batch_size : (i + 1) * batch_size].to(self.gpu)
            batch_seg = segment_ids[i * batch_size : (i + 1) * batch_size].to(self.gpu)
            batch_enb_mask = inp_enb_mask[i * batch_size : (i + 1) * batch_size].to(
                self.gpu
            )
            batch_ex_mask = inp_enb_mask[i * batch_size : (i + 1) * batch_size].to(
                self.gpu
            )

            batch_logits_enb = self.model(
                batch_enb, batch_seg[:, : inp_enb.size(1)], batch_enb_mask
            ).data.cpu()
            batch_logits_ex = self.model(
                batch_ex, batch_seg[:, : inp_ex.size(1)], batch_ex_mask
            ).data.cpu()

            logits_enb.append(batch_logits_enb)
            logits_ex.append(batch_logits_ex)

        logits_enb = torch.cat(logits_enb)
        logits_ex = torch.cat(logits_ex)
        #         logits_enb = self.model(inp_enb, segment_ids[:, :inp_enb.size(1)], inp_enb_mask)
        #         logits_ex = self.model(inp_ex, segment_ids[:, :inp_ex.size(1)], inp_ex_mask)

        contrib_logits = logits_enb - logits_ex  # [B * 2]
        if contrib_logits.size(1) == 2:
            contrib_score = contrib_logits[:, 1] - contrib_logits[:, 0]  # [B]
        else:
            contrib_score = normalize_logit(contrib_logits, label)
        contrib_score = contrib_score.mean()
        return contrib_score.item()

    def explain_single_transformer(
        self, input_ids, input_mask, segment_ids, region, label=None
    ):
        inp_flatten = input_ids.view(-1).cpu().numpy()
        inp_mask_flatten = input_mask.view(-1).cpu().numpy()
        if self.nb_method == "ngram":
            mask_regions = self.get_ngram_mask_region(region, inp_flatten)
        else:
            raise NotImplementedError("unknown method %s" % self.nb_method)

        score = self.score(
            inp_flatten, inp_mask_flatten, segment_ids, [region], mask_regions, label
        )
        return score
