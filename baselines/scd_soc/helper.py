import os
import torch
import numpy as np
from bert.run_classifier import BertConfig, BertForSequenceClassification
from algo.soc_transformer import SOCForTransformer
from algo.scd_transformer import CDForTransformer, SCDForTransformer
from utils.reader import (
    get_data_iterators_sst_flatten,
    get_data_iterators_yelp,
    get_data_iterators_tacred,
)
import sys

sys.path.append("../../src")
from application_utils.text_utils import prepare_huggingface_data


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_bert(bert_path, device):
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"

    output_model_file = os.path.join(bert_path, WEIGHTS_NAME)
    output_config_file = os.path.join(bert_path, CONFIG_NAME)

    config = BertConfig(output_config_file)
    model = BertForSequenceClassification(config, num_labels=2)
    model.load_state_dict(torch.load(output_model_file))
    model.eval()

    if device.index >= 0:
        model = model.to(device)
    return model


def get_lm_model(lm_path, gpu):
    lm_model = torch.load(
        lm_path, map_location=lambda storage, location: storage.cuda(gpu)
    )
    lm_model.gpu = gpu
    lm_model.encoder.gpu = gpu
    return lm_model


def get_hiexpl(
    method, bert_model, lm_path, tokenizer, device, sample_num=20, lm_model=None
):
    assert method in {"soc", "scd", "cd"}

    gpu = device.index

    class args:
        pass

    args.gpu = gpu
    args.task = "sst"
    args.dataset = "test"
    args.lm_path = lm_path
    args.nb_method = "ngram"
    args.d_out = 2
    # args.n_cells = args.n_layers
    args.use_gpu = args.gpu >= 0
    args.method = method
    args.nb_range = 10
    args.start = 0
    args.stop = 10
    args.batch_size = 1
    args.sample_n = sample_num
    args.use_bert_lm = True

    (
        text_field,
        length_field,
        train_iter,
        dev_iter,
        test_iter,
        train,
        dev,
    ) = get_data_iterators_sst_flatten(map_cpu=False)

    iter_map = {"train": train_iter, "dev": dev_iter, "test": test_iter}
    if args.task == "sst":
        tree_path = ".data/sst/trees/%s.txt"
    else:
        raise ValueError

    args.n_embed = len(text_field.vocab)

    if args.method == "soc":
        if lm_model is None:
            lm_model = get_lm_model(args.lm_path, args.gpu)
        algo = SOCForTransformer(
            bert_model,
            lm_model,
            tree_path=tree_path % args.dataset,
            output_path=None,
            #                                  output_path='outputs/' + args.task + '/soc_bert_results/soc%s.txt' % args.exp_name,
            config=args,
            vocab=text_field.vocab,
            tokenizer=tokenizer,
        )
    elif args.method == "scd":
        if lm_model is None:
            lm_model = get_lm_model(args.lm_path, args.gpu)
        algo = SCDForTransformer(
            bert_model,
            lm_model,
            tree_path=tree_path % args.dataset,
            #                                  output_path='outputs/' + args.task + '/scd_bert_results/scd%s.txt' % args.exp_name,
            output_path=None,
            config=args,
            vocab=text_field.vocab,
            tokenizer=tokenizer,
        )

    elif args.method == "cd":
        algo = CDForTransformer(
            bert_model,
            tree_path=tree_path % args.dataset,
            #                                  output_path='outputs/' + args.task + '/scd_bert_results/scd%s.txt' % args.exp_name,
            output_path=None,
            config=args,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError

    return algo


def get_prediction(model, sentences, tokenizer, device):
    X = prepare_huggingface_data(sentences, tokenizer)
    for key in X:
        X[key] = torch.from_numpy(X[key]).to(device)
    preds = model(
        X["input_ids"], X["token_type_ids"].long(), X["attention_mask"].long()
    )
    return preds


def explain_sentence(sentence, algo, tokenizer, spans=None):

    X = prepare_huggingface_data([sentence], tokenizer)

    for key in X:
        X[key] = torch.from_numpy(X[key]).to(torch.device("cuda:" + str(algo.gpu)))

    inp = X["input_ids"]
    segment_ids = X["token_type_ids"].long()
    input_mask = X["attention_mask"].long()

    if spans is None:
        spans = [(x, x) for x in range(0, inp.shape[1] - 2)]

    contribs = {}
    for span in spans:
        span = (span[0] + 1, span[1] + 1)
        contrib = algo.explain_single_transformer(inp, input_mask, segment_ids, span)
        contribs[span] = contrib

    tokens = tokenizer.convert_ids_to_tokens(inp.view(-1).cpu().numpy())

    return contribs, tokens[1:-1]
