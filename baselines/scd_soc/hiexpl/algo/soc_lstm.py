import torch
from utils.args import get_args
import copy
from utils.parser import read_trees_from_corpus, get_span_to_node_mapping
import pickle
from utils.agglomeration import *
from skimage import measure
import os

args = get_args()


def append_extra_input(batch, extra_input_dic):
    batch_size = batch.text.size(1)
    seq_len = batch.text.size(0)
    for k in extra_input_dic:
        if not k.endswith("label"):
            v = extra_input_dic[k].cuda()
            if v.size(0) < batch.text.size(0):
                pv = torch.zeros(batch.text.size(0), v.size(1)).cuda()
                pv[: v.size(0)] = v
                v = pv
            else:
                v = v[: batch.text.size(0)]
            v = v.expand(-1, batch_size).cuda()
            # mask tokens, according to the input batch
            pad_position = torch.eq(batch.text, 1)
            v.masked_fill_(pad_position, 1)
            setattr(batch, k, v)


def normalize_logit(contrib_logits, gt_label):
    # [B, L]
    if not args.class_score:
        contrib_logits_bak = contrib_logits.clone()
        contrib_logits_bak[:, gt_label] = -1000
        second_max, _ = contrib_logits_bak.max(-1)
        scores = contrib_logits[:, gt_label] - second_max
    else:
        scores = contrib_logits[:, gt_label]
    return scores


class Batch:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class ExplanationBase:
    def __init__(
        self,
        target_model,
        data_iterator,
        vocab,
        tree_path,
        output_path,
        config,
        pad_idx=1,
    ):
        self.model = target_model
        self.iterator = data_iterator
        self.pad_idx = pad_idx
        if args.task == "sst":
            self.trees = read_trees_from_corpus(tree_path)
        self.nb_method = config.nb_method
        self.nb_unidirectional = False
        self.nb_range = config.nb_range
        self.gpu = config.gpu

        self.vocab = vocab
        self.output_path = output_path
        if self.output_path is not None:
            os.makedirs("/".join(self.output_path.split("/")[:-1]), exist_ok=True)

        self.batch_start = config.start
        self.batch_stop = config.stop

        self.label_vocab = None

    def decode(self, l):
        return " ".join([self.vocab.itos[x] for x in l])

    def encode(self, l):
        return [self.vocab.stoi[x] for x in l]

    def mask_region(self, inp, region_indicator, mask_value):
        new_seq = []
        for i in range(len(region_indicator)):
            if region_indicator[i] not in mask_value:
                new_seq.append(inp[i])
            else:
                new_seq.append(1)
        new_seq = np.array(new_seq)
        return new_seq

    def get_ngram_mask_region(self, region, inp):
        if not self.nb_unidirectional:
            return [
                (
                    max(region[0] - self.nb_range, 0),
                    min(region[1] + self.nb_range, len(inp) - 1),
                )
            ]
        else:
            return [(region[0], min(region[1] + self.nb_range, len(inp) - 1))]

    def explain_single(self, inp, inp_id, region, extra_input=None):
        pass

    def repr_result_word(self, inp, contribs):
        tokens = [self.vocab.itos[inp[i]] for i in range(len(inp))]
        output_str = " ".join(
            ["%s %.6f\t" % (tk, sc) for (tk, sc) in zip(tokens, contribs)]
        )
        return output_str

    #     def explain_sst_word(self):
    #         f = open(self.output_path, 'w')
    #         all_contribs = []
    #         for batch_idx, batch in enumerate(self.iterator):
    #             inp = batch.text.view(-1).cpu().numpy()
    #             inp_id = batch.offset.item()
    #             contribs = []
    #             for i in range(len(inp)):
    #                 contrib = self.explain_single(inp, inp_id, i)
    #                 contribs.append(contrib)
    #             all_contribs.append(contribs)
    #             s = self.repr_result_word(inp, contribs)
    #             f.write(s + '\n')
    #             print('finished %d' % batch_idx)
    #         f.close()
    #         return all_contribs

    def repr_result_region(self, inp, spans, contribs, gt_label=None):
        tokens = [self.vocab.itos[inp[i]] for i in range(len(inp))]
        outputs = []
        assert len(spans) == len(contribs)
        for span, contrib in zip(spans, contribs):
            if type(span) is tuple:
                outputs.append((" ".join(tokens[span[0] : span[1] + 1]), contrib))
            else:
                outputs.append((tokens[span], contrib))
        output_str = " ".join(["%s %.6f\t" % (x, y) for x, y in outputs])
        if gt_label is not None:
            output_str = self.label_vocab.itos[gt_label] + "\t" + output_str

        return output_str

    def explain_sst(self):
        f = open(self.output_path, "w")
        all_contribs = []
        cnt = 0
        for batch_idx, batch in enumerate(self.iterator):
            if batch_idx < self.batch_start:
                continue
            inp = batch.text.view(-1).cpu().numpy()
            inp_id = batch.offset.item()

            #             print(batch.offset, inp_id)

            span2node, node2span = get_span_to_node_mapping(self.trees[inp_id])
            spans = list(span2node.keys())

            if args.no_subtrees:
                spans = spans[-1:]

            contribs = []

            for span in spans:
                if type(span) is int:
                    span = (span, span)
                #                 print("inp", inp, inp.shape, inp_id, "span", span)
                contrib = self.explain_single(inp, inp_id, span)
                contribs.append(contrib)
            all_contribs.append(contribs)

            s = self.repr_result_region(inp, spans, contribs)
            f.write(s + "\n")

            print("finished %d" % batch_idx)
            cnt += 1
            if batch_idx == self.batch_stop - 1:
                break
        f.close()
        return all_contribs


#     def explain_token(self, dataset):
#         f = open(self.output_path, 'w')
#         all_contribs = []
#         avg_score = 0
#         cnt = 0
#         extra_input = {}
#         for batch_idx, batch in enumerate(self.iterator):
#             if batch_idx < self.batch_start:
#                 continue
#             inp = batch.text.view(-1).cpu().numpy()
#             inp_id = batch.offset.item()
#             label = None
#             if dataset == 'tacred':
#                 if batch.label.item() == 0:
#                     continue
#                 pred_logits = self.model(batch)
#                 _, pred = torch.max(pred_logits, -1)
#                 if pred.item() != batch.label.item():
#                     continue
#                 label = extra_input['gt_label'] = batch.label
#                 extra_input['pos'] = batch.pos
#                 extra_input['ner'] = batch.ner

#             spans = range(0, inp.shape[0])
#             contribs = []

#             for span in spans:
#                 if type(span) is int:
#                     span = (span, span)
#                     contrib = self.explain_single(inp, inp_id, span, extra_input=extra_input)
#                 contribs.append(contrib)
#             all_contribs.append(contribs)

#             s = self.repr_result_region(inp, spans, contribs, gt_label=label.item())
#             f.write(s + '\n')

#             print('finished %d' % batch_idx)
#             cnt += 1
#             if batch_idx >= self.batch_stop - 1:
#                 break
#         avg_score /= cnt
#         print(avg_score)
#         f.close()
#         return all_contribs

#     def explain_agg(self, dataset):
#         f = open(self.output_path, 'wb')
#         all_tabs = []

#         for batch_idx, batch in enumerate(self.iterator):
#             if batch_idx < self.batch_start:
#                 continue
#             if dataset == 'tacred':
#                 if batch.label.item() == 0:
#                     continue

#                 pred_logits = self.model(batch)
#                 _, pred = torch.max(pred_logits, -1)
#                 if pred.item() != batch.label.item():
#                     continue
#             # note: method='cd' has no effect
#             lists = self.agglomerate(batch, percentile_include=90, method='cd', sweep_dim=1, dataset=dataset,
#                                      num_iters=10)
#             lists = collapse_tree(lists)
#             data = lists_to_tabs(lists, batch.text.size(0))
#             text = self.decode(batch.text.view(-1).cpu().numpy())
#             if self.label_vocab is not None:
#                 label_name = self.label_vocab.itos[batch.label.item()]
#             else:
#                 label_name = ''
#             all_tabs.append({
#                 'tab': data,
#                 'text': text,
#                 'label': label_name
#             })
#             print('finished %d' % batch_idx)

#             if batch_idx >= self.batch_stop - 1:
#                 break
#         pickle.dump(all_tabs, f)
#         f.close()
#         return all_tabs

#     def agglomerate(self, batch, percentile_include, method, sweep_dim, dataset,
#                     num_iters=5, subtract=True, absolute=True):
#         # MIT License
#         #
#         # Copyright (c) 2019 Chandan Singh
#         #
#         # Permission is hereby granted, free of charge, to any person obtaining a copy
#         # of this software and associated documentation files (the "Software"), to deal
#         # in the Software without restriction, including without limitation the rights
#         # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#         # copies of the Software, and to permit persons to whom the Software is
#         # furnished to do so, subject to the following conditions:
#         #
#         # The above copyright notice and this permission notice shall be included in all
#         # copies or substantial portions of the Software.

#         # Modified from https://github.com/csinva/hierarchical-dnn-interpretations/
#         extra_input = {}
#         label = None
#         if dataset == 'tacred':
#             if batch.label.item() == 0:
#                 return None
#             label = extra_input['gt_label'] = batch.label
#             extra_input['pos'] = batch.pos
#             extra_input['ner'] = batch.ner

#         # get original text and score
#         text_orig = batch.text.data.cpu().numpy() # [T, 1]
#         for t in range(text_orig.shape[0]):
#             if text_orig[t] == 0:
#                 text_orig[t] = 1
#         score_orig = self.explain_single(text_orig.reshape(-1), None, [0, text_orig.shape[0] - 1], extra_input)


#         # get scores
#         texts = gen_tiles(text_orig, method=method, sweep_dim=sweep_dim)
#         texts = texts.transpose()

#         starts, stops = tiles_to_cd(texts)
#         scores = []
#         for (start, stop) in zip(starts, stops):
#             score = self.explain_single(text_orig.reshape(-1), None, [start, stop], extra_input)
#             scores.append(score)

#         # threshold scores
#         mask = threshold_scores(scores, percentile_include, absolute=absolute)

#         # initialize lists
#         scores_list = [np.copy(scores)]
#         mask_list = [mask]
#         comps_list = []
#         comp_scores_list = [{0: score_orig}]

#         # iterate
#         for step in range(num_iters):
#             # find connected components for regions
#             comps = np.copy(measure.label(mask_list[-1], background=0, connectivity=1))

#             # loop over components
#             comp_scores_dict = {}
#             for comp_num in range(1, np.max(comps) + 1):

#                 # make component tile
#                 comp_tile_bool = (comps == comp_num)
#                 comp_tile = gen_tile_from_comp(text_orig, comp_tile_bool, method)

#                 # make tiles around component
#                 border_tiles = gen_tiles_around_baseline(text_orig, comp_tile_bool,
#                                                                 method=method,
#                                                                 sweep_dim=sweep_dim)

#                 # predict for all tiles
#                 # format tiles into batch
#                 tiles_concat = np.hstack((comp_tile, np.squeeze(border_tiles[0]).transpose()))

#                 starts, stops = tiles_to_cd(tiles_concat)
#                 scores_all = []
#                 for (start, stop) in zip(starts, stops):
#                     score = self.explain_single(text_orig.reshape(-1), None, [start, stop], extra_input)
#                     scores_all.append(score)

#                 score_comp = np.copy(scores_all[0])
#                 scores_border_tiles = np.copy(scores_all[1:])

#                 # store the predicted class scores
#                 comp_scores_dict[comp_num] = np.copy(score_comp)

#                 # update pixel scores
#                 tiles_idxs = border_tiles[1]
#                 for i, idx in enumerate(tiles_idxs):
#                     scores[idx] = scores_border_tiles[i] - score_comp

#             # get class preds and thresholded image
#             scores = np.array(scores)
#             scores[mask_list[-1]] = np.nan
#             mask = threshold_scores(scores, percentile_include, absolute=absolute)

#             # add to lists
#             scores_list.append(np.copy(scores))
#             mask_list.append(mask_list[-1] + mask)
#             comps_list.append(comps)
#             comp_scores_list.append(comp_scores_dict)

#             if np.sum(mask) == 0:
#                 break

#         # pad first image
#         comps_list = [np.zeros(text_orig.size, dtype=np.int)] + comps_list

#         return {'scores_list': scores_list,  # arrs of scores (nan for selected)
#                 'mask_list': mask_list,  # boolean arrs of selected
#                 'comps_list': comps_list,  # arrs of comps with diff number for each comp
#                 'comp_scores_list': comp_scores_list,  # dicts with score for each comp
#                 'score_orig': score_orig}  # original score

#     def demo(self):
#         f = open(self.output_path, 'w')
#         while True:
#             l = input('sentence?')
#             inp_word = l.strip().split()
#             inp_word_id = [self.vocab.stoi.get(x,'<unk>') for x in inp_word]
#             inp = np.array(inp_word_id, dtype=np.int32)
#             spans = [(x,x) for x in range(len(inp_word_id))]

#             contribs = []
#             for span in spans:
#                 contrib = self.explain_single(inp, None, span)
#                 contribs.append(contrib)

#             s = self.repr_result_region(inp, spans, contribs)
#             f.write(s + '\n')
#             print(s)


class SOCForLSTM(ExplanationBase):
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
            target_model, data_iterator, vocab, tree_path, output_path, config, pad_idx
        )
        self.lm_model = lm_model
        self.sample_num = config.sample_n
        self.use_bert_lm = config.use_bert_lm

    def score(self, inp, x_regions, nb_regions, extra_input):
        # suppose only have one x_region and one nb_region
        x_region = x_regions[0]
        nb_region = nb_regions[0]

        inp_lm = copy.copy(inp)
        for i in range(len(inp_lm)):
            if (
                nb_region[0] <= i <= nb_region[1]
                and not x_region[0] <= i <= x_region[1]
            ):
                inp_lm[i] = 1

        inp_th = torch.from_numpy(inp_lm).long().view(-1, 1)
        inp_length = torch.LongTensor([len(inp)])
        fw_pos = torch.LongTensor([min(x_region[1] + 1, len(inp))])
        bw_pos = torch.LongTensor([max(x_region[0] - 1, -1)])

        if self.gpu >= 0:
            inp_th = inp_th.to(self.gpu)
            inp_length = inp_length.to(self.gpu)
            fw_pos = fw_pos.to(self.gpu)
            bw_pos = bw_pos.to(self.gpu)

        batch = Batch(text=inp_th, length=inp_length, fw_pos=fw_pos, bw_pos=bw_pos)

        inp_enb, inp_ex = [], []
        max_sample_length = self.nb_range + 1
        if self.sample_num > 0:

            # sample token ids..
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
            # insert token id samples into spaces before and after target phrase ids
            if len_bw > 0:
                filled_inp[nb_region[0] : x_region[0]] = bw_sample_seq[-len_bw:]
            if len_fw > 0:
                filled_inp[x_region[1] + 1 : nb_region[1] + 1] = fw_sample_seq[:len_fw]

            filled_ex = []
            for i in range(len(inp)):
                if not x_region[0] <= i <= x_region[1]:
                    filled_ex.append(filled_inp[i])
                else:
                    # make sure that the phrase is left empty (set to 1)
                    filled_ex.append(1)
            filled_ex = np.array(filled_ex, dtype=np.int32)

            # the sampled sentnece with the phrase (input) still there
            inp_enb.append(filled_inp)
            # the sampled sentnece with the phrase "zeroed"
            inp_ex.append(filled_ex)

        if args.add_itself:
            filled_inp = inp
            filled_ex = []
            for i in range(len(inp)):
                if not x_region[0] <= i <= x_region[1]:
                    filled_ex.append(filled_inp[i])
                else:
                    filled_ex.append(1)
            filled_ex = np.array(filled_ex, dtype=np.int32)
            # the original sentence without modification
            inp_enb.append(filled_inp)
            # the original sentence with the phrase zerod out
            inp_ex.append(filled_ex)

        inp_enb, inp_ex = np.stack(inp_enb), np.stack(inp_ex)
        inp_enb, inp_ex = (
            torch.from_numpy(inp_enb).long(),
            torch.from_numpy(inp_ex).long(),
        )
        if self.gpu >= 0:
            inp_enb, inp_ex = inp_enb.to(self.gpu), inp_ex.to(self.gpu)
        inp_enb, inp_ex = inp_enb.transpose(0, 1), inp_ex.transpose(0, 1)
        batch_enb = Batch(text=inp_enb)
        batch_ex = Batch(text=inp_ex)

        if extra_input is not None:
            append_extra_input(batch_enb, extra_input)
            append_extra_input(batch_ex, extra_input)

        logits_nb, logits_ex = self.model(batch_enb), self.model(batch_ex)

        contrib_logits = logits_nb - logits_ex  # [B * 2]
        if contrib_logits.size(1) == 2:  # 2-way classifier

            ## what is going on here (accumulate the result of each logit)
            contrib_score = contrib_logits[:, 0] - contrib_logits[:, 1]  # [B]
        else:
            gt_label = extra_input["gt_label"]
            # margin from no_relation
            contrib_score = normalize_logit(contrib_logits, gt_label)

        # average over the samples (the expectation)
        contrib_score = contrib_score.mean()
        return contrib_score.item()

    def explain_single(self, inp, inp_id, region, extra_input=None):
        """
        :param region: the input region to be explained
        :param inp: numpy array
        :param inp_id: int
        :return:
        """

        mask_regions = self.get_ngram_mask_region(region, inp)

        print(
            "inp",
            inp,
            "region",
            [region],
            "mask region",
            mask_regions,
            "extra input",
            extra_input,
        )
        score = self.score(inp, [region], mask_regions, extra_input)

        return score
