from algo.soc_lstm import SOCForLSTM
from algo.scd_lstm import CDForLSTM, SCDForLSTM
from algo.soc_transformer import SOCForTransformer
from algo.scd_transformer import CDForTransformer, SCDForTransformer
import torch
import argparse
from utils.args import get_args
from utils.reader import (
    get_data_iterators_sst_flatten,
    get_data_iterators_yelp,
    get_data_iterators_tacred,
)
import random, os
from bert.run_classifier import BertConfig, BertForSequenceClassification
from nns.model import LSTMMeanRE, LSTMMeanSentiment, LSTMSentiment


def get_args_exp():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method")
    args = parser.parse_args()
    return args


args = get_args()

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    if args.task == "sst" or args.task == "sst_async":
        (
            text_field,
            length_field,
            train_iter,
            dev_iter,
            test_iter,
            train,
            dev,
        ) = get_data_iterators_sst_flatten(map_cpu=False)
    elif args.task == "yelp":
        (
            text_field,
            label_field,
            train_iter,
            dev_iter,
            test_iter,
            train,
            dev,
        ) = get_data_iterators_yelp(map_cpu=False)
    elif args.task == "tacred":
        (
            text_field,
            label_field,
            subj_offset,
            obj_offset,
            pos,
            ner,
            train_iter,
            dev_iter,
            test_iter,
            train,
            dev,
        ) = get_data_iterators_tacred()
    else:
        raise ValueError("unknown task")

    iter_map = {"train": train_iter, "dev": dev_iter, "test": test_iter}
    if args.task == "sst":
        tree_path = ".data/sst/trees/%s.txt"
    elif args.task == "yelp":
        tree_path = ".data/yelp_review_polarity_csv/%s.csv"
    elif args.task == "tacred":
        tree_path = ".data/TACRED/data/json/%s.json"
    else:
        raise ValueError

    if args.task == "tacred":
        args.label_vocab = label_field.vocab

    args.n_embed = len(text_field.vocab)
    args.d_out = 2 if args.task in ["sst", "yelp"] else len(label_field.vocab)
    args.n_cells = args.n_layers
    args.use_gpu = args.gpu >= 0

    if args.explain_model == "lstm":
        cls = {"sst": LSTMSentiment, "yelp": LSTMMeanSentiment, "tacred": LSTMMeanRE}
        model = cls[args.task](args)
        model.load_state_dict(torch.load(args.resume_snapshot))
        model = model.to(args.gpu)
        model.gpu = args.gpu
        model.use_gpu = args.gpu >= 0
        model.eval()
        print(model)
        if args.method == "soc":
            lm_model = torch.load(
                args.lm_path,
                map_location=lambda storage, location: storage.cuda(args.gpu),
            )
            lm_model.gpu = args.gpu
            lm_model.encoder.gpu = args.gpu
            algo = SOCForLSTM(
                model,
                lm_model,
                iter_map[args.dataset],
                tree_path=tree_path % args.dataset,
                config=args,
                vocab=text_field.vocab,
                output_path="outputs/"
                + args.task
                + "/soc_results/soc%s.txt" % args.exp_name,
            )
        elif args.method == "scd":
            lm_model = torch.load(
                args.lm_path,
                map_location=lambda storage, location: storage.cuda(args.gpu),
            )
            lm_model.gpu = args.gpu
            lm_model.encoder.gpu = args.gpu
            algo = SCDForLSTM(
                model,
                lm_model,
                iter_map[args.dataset],
                tree_path=tree_path % args.dataset,
                config=args,
                vocab=text_field.vocab,
                output_path="outputs/"
                + args.task
                + "/scd_results/scd%s.txt" % args.exp_name,
            )
        else:
            raise ValueError("unknown method")
    elif args.explain_model == "bert":
        CONFIG_NAME = "bert_config.json"
        WEIGHTS_NAME = "pytorch_model.bin"
        output_model_file = os.path.join("bert/%s" % args.resume_snapshot, WEIGHTS_NAME)
        output_config_file = os.path.join("bert/%s" % args.resume_snapshot, CONFIG_NAME)
        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        model = BertForSequenceClassification(
            config, num_labels=2 if args.task != "tacred" else 42
        )
        model.load_state_dict(torch.load(output_model_file))
        model.eval()
        if args.gpu >= 0:
            model = model.to(args.gpu)
        if args.method == "soc":
            lm_model = torch.load(
                args.lm_path,
                map_location=lambda storage, location: storage.cuda(args.gpu),
            )
            lm_model.gpu = args.gpu
            lm_model.encoder.gpu = args.gpu
            algo = SOCForTransformer(
                model,
                lm_model,
                tree_path=tree_path % args.dataset,
                output_path="outputs/"
                + args.task
                + "/soc_bert_results/soc%s.txt" % args.exp_name,
                config=args,
                vocab=text_field.vocab,
            )
        elif args.method == "scd":
            lm_model = torch.load(
                args.lm_path,
                map_location=lambda storage, location: storage.cuda(args.gpu),
            )
            lm_model.gpu = args.gpu
            lm_model.encoder.gpu = args.gpu
            algo = SCDForTransformer(
                model,
                lm_model,
                tree_path=tree_path % args.dataset,
                output_path="outputs/"
                + args.task
                + "/scd_bert_results/scd%s.txt" % args.exp_name,
                config=args,
                vocab=text_field.vocab,
            )
        else:
            raise ValueError("unknown method")
    else:
        raise ValueError("unknown model")
    with torch.no_grad():
        if args.task == "sst":
            with torch.cuda.device(args.gpu):
                if args.agg:
                    algo.explain_agg("sst")
                else:
                    algo.explain_sst()
        elif args.task == "yelp":
            with torch.cuda.device(args.gpu):
                if args.agg:
                    algo.explain_agg("sst")
                algo.explain_token("yelp")
        elif args.task == "tacred":
            with torch.cuda.device(args.gpu):
                algo.label_vocab = label_field.vocab
                algo.ner_vocab = ner.vocab
                algo.pos_vocab = pos.vocab
                if args.agg:
                    algo.explain_agg("tacred")
                else:
                    algo.explain_token("tacred")
