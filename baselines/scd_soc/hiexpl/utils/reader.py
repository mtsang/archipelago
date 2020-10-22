import torchtext as tt
from nltk import Tree
import pickle, random
import torch
from utils.args import get_args, makedirs
import os
from bert.tokenization import BertTokenizer
import csv, json

args = get_args()


def save_vocab(path, vocab):
    f = open(path, "wb")
    pickle.dump(vocab, f)
    f.close()


def load_vocab(path):
    f = open(path, "rb")
    obj = pickle.load(f)
    return obj


def handle_vocab(
    vocab_path, field, datasets, vector_cache="", train_lm=False, max_size=None
):
    create_vocab = False
    if os.path.isfile(vocab_path):
        print("loading vocab from %s" % vocab_path)
        vocab = load_vocab(vocab_path)
        field.vocab = vocab
    else:
        print("creating vocab")
        makedirs("vocab")
        field.build_vocab(*datasets, max_size=max_size)
        vocab = field.vocab
        if "<s>" not in field.vocab.stoi:
            field.vocab.itos.append("<s>")
            field.vocab.stoi["<s>"] = len(field.vocab.itos) - 1
        if "</s>" not in field.vocab.stoi:
            field.vocab.itos.append("</s>")
            field.vocab.stoi["</s>"] = len(field.vocab.itos) - 1
        save_vocab(vocab_path, vocab)
        create_vocab = True

    if vector_cache != "" and not vector_cache.startswith("none"):
        if args.word_vectors or create_vocab:
            if os.path.isfile(vector_cache):
                field.vocab.vectors = torch.load(vector_cache)
            else:
                field.vocab.load_vectors(args.word_vectors)
                for i in range(field.vocab.vectors.size(0)):
                    if (
                        field.vocab.vectors[i, 0].item() == 0
                        and field.vocab.vectors[i, 1].item() == 0
                    ):
                        field.vocab.vectors[i].uniform_(-1, 1)
                makedirs(os.path.dirname(vector_cache))
                torch.save(field.vocab.vectors, vector_cache)

    if train_lm:
        v = torch.zeros(2, field.vocab.vectors.size(-1))
        field.vocab.vectors = torch.cat([field.vocab.vectors, v], 0)


def get_data_iterators_sst():
    inputs = tt.data.Field(lower=args.lower)
    answers = tt.data.Field(sequential=False, unk_token=None)
    train, dev, test = tt.datasets.SST.splits(
        inputs,
        answers,
        fine_grained=False,
        train_subtrees=not args.no_subtrees,
        filter_pred=lambda ex: ex.label != "neutral",
    )

    vocab_path = (
        "vocab/vocab_sst.pkl"
        if not args.use_bert_tokenizer
        else "vocab/vocab_sst_bert.pkl"
    )
    if args.fix_test_vocab and not args.use_bert_tokenizer:
        vocab_path = "vocab/vocab_sst_fix.pkl"
    c_postfix = ".sst"
    if args.use_bert_tokenizer:
        c_postfix += ".bert"
    if args.fix_test_vocab:
        c_postfix += ".fix"
    handle_vocab(vocab_path, inputs, (train, dev, test), args.vector_cache + c_postfix)
    answers.build_vocab(train)

    if args.include_noise_labels:
        train_2, _, _ = tt.datasets.SST.splits(
            inputs, answers, fine_grained=False, train_subtrees=True
        )
        texts = set()
        for example in train_2.examples:
            if len(example.text) == 1 and example.text[0] not in texts:
                texts.add(example.text[0])
                if example.label == "positive":
                    example.label = "negative"
                elif example.label == "negative":
                    example.label = "positive"
                elif example.label == "neutral":
                    example.label = random.choice(["positive", "negative"])
                else:
                    raise ValueError
                train.examples.append(example)
        train_iter, dev_iter, test_iter = tt.data.BucketIterator.splits(
            (train, dev, test),
            batch_size=args.batch_size,
            device=torch.device(args.gpu),
            sort=True,
            shuffle=False,
        )
    else:
        train_iter, dev_iter, test_iter = tt.data.BucketIterator.splits(
            (train, dev, test),
            batch_size=args.batch_size,
            device=torch.device(args.gpu),
        )
    return inputs, answers, train_iter, dev_iter, test_iter, train, dev


def compute_mapping(tokens, bert_tokens):
    mapping = []
    i, j = 0, 0
    while i < len(tokens):
        t = ""
        while len(t) < len(tokens[i]):
            t += bert_tokens[j].replace("##", "")
            j += 1
        if len(t) > len(tokens[i]):
            print("warning: mapping mismatch")
            break
        mapping.append(j)
        i += 1
    return mapping


def convert_to_bert_tokenization(tokens, bert_tokenizer, return_mapping=False):
    text = " ".join(tokens)
    bert_tokens = bert_tokenizer.tokenize(text)
    # compute mapping
    if return_mapping:
        mapping = compute_mapping(tokens, bert_tokens)
        return bert_tokens, mapping
    else:
        return bert_tokens


def get_examples_sst(path, train_lm, bert_tokenizer=None):
    f = open(path)
    examples = []
    for i, line in enumerate(f.readlines()):
        line = line.lower()
        tree = Tree.fromstring(line)

        tokens = tree.leaves()
        if bert_tokenizer is not None:
            tokens, mapping = convert_to_bert_tokenization(
                tokens, bert_tokenizer, return_mapping=True
            )

        if train_lm:
            tokens = ["<s>"] + tokens + ["</s>"]

        example = tt.data.Example()
        example.text = tokens
        example.length = len(tokens)
        examples.append(example)
        example.offset = i

        if bert_tokenizer is not None:
            example.mapping = mapping

        if int(tree.label()) >= 3:
            example.label = 0
        elif int(tree.label()) <= 2:
            example.label = 1
        else:
            example.label = 2

    return examples


def get_examples_yelp(path, train_lm, bert_tokenizer=None):
    f = open(path)
    reader = csv.reader(f)
    examples = []
    for i, line in enumerate(reader):
        tokens = line[1].split()
        label = int(line[0])

        if bert_tokenizer is not None:
            tokens, mapping = convert_to_bert_tokenization(
                tokens[:100], bert_tokenizer, return_mapping=True
            )

        if train_lm:
            tokens = ["<s>"] + tokens[:50] + ["</s>"]

        if args.filter_length_gt != -1 and len(tokens) >= args.filter_length_gt:
            continue

        example = tt.data.Example()
        example.text = tokens
        example.length = len(tokens)
        example.offset = i

        if bert_tokenizer is not None:
            example.mapping = mapping

        if label == 2:
            example.label = 0
        elif label == 1:
            example.label = 1
        else:
            raise ValueError

        examples.append(example)
        if args.explain_model == "bert":
            if i == args.stop:
                break

    return examples


def filter_tacred_special_token(entry):
    text = entry["token"]
    text[entry["subj_start"]] = "SUBJ%s" % entry["subj_type"]
    text[entry["obj_start"]] = "OBJ %s" % entry["obj_type"]
    for idx in range(entry["subj_start"] + 1, entry["subj_end"] + 1):
        text[idx] = "-!REMOVED!-"

    for idx in range(entry["obj_start"] + 1, entry["obj_end"] + 1):
        text[idx] = "-!REMOVED!-"
    text = [_ for _ in filter(lambda x: x != "-!REMOVED!-", text)]
    full_tokens = []
    for token in text:
        full_tokens.extend(token.split())
    return full_tokens


def get_examples_tacred(path, train_lm, bert_tokenizer=None):
    f = open(path)
    # reader = csv.reader(f)
    examples = []
    json_data = json.load(f)

    for ji, entry in enumerate(json_data):
        tokens = entry["token"]
        # tokens = [_.lower() for _ in tokens]
        if bert_tokenizer is not None:
            tokens = filter_tacred_special_token(entry)
            tokens = [_.lower() for _ in tokens]
            tokens, mapping = convert_to_bert_tokenization(
                tokens, bert_tokenizer, return_mapping=True
            )

        if args.filter_length_gt != -1 and len(tokens) >= args.filter_length_gt:
            continue

        example = tt.data.Example()
        example.text = tokens
        example.label = entry["relation"]
        example.offset = ji

        example.subj_offset, example.obj_offset = [], []
        example.pos, example.ner = entry["stanford_pos"], entry["stanford_ner"]

        if bert_tokenizer is None:
            for idx in range(entry["subj_start"], entry["subj_end"] + 1):
                example.text[idx] = "SUBJ-%s" % entry["subj_type"]
            for idx in range(entry["obj_start"], entry["obj_end"] + 1):
                example.text[idx] = "OBJ-%s" % entry["obj_type"]

        for idx in range(len(tokens)):
            if idx < entry["subj_start"]:
                example.subj_offset.append(idx - entry["subj_start"])
            elif idx > entry["subj_end"]:
                example.subj_offset.append(idx - entry["subj_end"])
            else:
                example.subj_offset.append(0)
            if idx < entry["obj_start"]:
                example.obj_offset.append(idx - entry["obj_start"])
            elif idx > entry["obj_end"]:
                example.obj_offset.append(idx - entry["obj_end"])
            else:
                example.obj_offset.append(0)

        if bert_tokenizer is not None:
            example.mapping = mapping

        if train_lm:
            example.text = ["<s>"] + example.text[:50] + ["</s>"]
        example.length = len(example.text)
        examples.append(example)
        if args.explain_model == "bert":
            if ji == args.stop:
                break

    return examples


def get_data_iterators_sst_flatten(train_lm=False, map_cpu=False):
    text_field = tt.data.Field(lower=args.lower)
    length_field = tt.data.Field(sequential=False, use_vocab=False)
    offset_field = tt.data.Field(sequential=False, use_vocab=False)
    _, _, _ = tt.datasets.SST.splits(
        text_field,
        length_field,
        fine_grained=False,
        train_subtrees=False,
        filter_pred=lambda ex: ex.label != "neutral",
    )

    path_format = "./.data/sst/trees/%s.txt"

    bert_tokenizer = None
    if args.use_bert_tokenizer:
        bert_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True, cache_dir="bert/cache"
        )

    train_ex, dev_ex, test_ex = (
        get_examples_sst(path_format % ds, train_lm, bert_tokenizer=bert_tokenizer)
        for ds in ["train", "dev", "test"]
    )
    train, dev, test = (
        tt.data.Dataset(
            ex,
            [("text", text_field), ("length", length_field), ("offset", offset_field)],
        )
        for ex in [train_ex, dev_ex, test_ex]
    )

    vocab_path = (
        "vocab/vocab_sst.pkl"
        if not args.use_bert_tokenizer
        else "vocab/vocab_sst_bert.pkl"
    )
    c_postfix = ".sst"
    if args.use_bert_tokenizer:
        c_postfix += ".bert"
    handle_vocab(
        vocab_path,
        text_field,
        (train, dev, test),
        args.vector_cache + c_postfix,
        train_lm,
    )

    train_iter, dev_iter, test_iter = (
        tt.data.BucketIterator(
            x,
            batch_size=args.batch_size,
            device=torch.device(args.gpu) if not map_cpu else "cpu",
            shuffle=False,
        )
        for x in (train, dev, test)
    )
    return text_field, length_field, train_iter, dev_iter, test_iter, train, dev


def get_data_iterators_yelp(train_lm=False, map_cpu=False):
    text_field = tt.data.Field(lower=args.lower)
    label_field = tt.data.LabelField(sequential=False, unk_token=None)
    length_field = tt.data.Field(sequential=False, use_vocab=False)
    offset_field = tt.data.Field(sequential=False, use_vocab=False)

    path_format = "./.data/yelp_review_polarity_csv/%s.csv.token"
    bert_tokenizer = None
    if args.use_bert_tokenizer:
        bert_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True, cache_dir="bert/cache"
        )
    train_examples, test_examples = (
        get_examples_yelp(path_format % ds, train_lm, bert_tokenizer=bert_tokenizer)
        for ds in ["train", "test"]
    )
    dev_examples = test_examples[:500]
    train, dev, test = (
        tt.data.Dataset(
            ex,
            [
                ("text", text_field),
                ("length", length_field),
                ("offset", offset_field),
                ("label", label_field),
            ],
        )
        for ex in [train_examples, dev_examples, test_examples]
    )

    vocab_path = (
        "vocab/vocab_yelp.pkl"
        if not args.use_bert_tokenizer
        else "vocab/vocab_yelp_bert.pkl"
    )
    if args.fix_test_vocab and not args.use_bert_tokenizer:
        vocab_path = "vocab/vocab_yelp_fix.pkl"

    c_postfix = ".yelp"
    if args.use_bert_tokenizer:
        c_postfix += ".bert"
    if args.fix_test_vocab:
        c_postfix += ".fix"
    handle_vocab(
        vocab_path,
        text_field,
        (train, test),
        args.vector_cache + c_postfix,
        train_lm,
        max_size=20000,
    )
    label_field.build_vocab(train)
    train_iter, dev_iter, test_iter = (
        tt.data.BucketIterator(
            x,
            batch_size=args.batch_size,
            device=torch.device(args.gpu) if not map_cpu else "cpu",
            shuffle=False,
        )
        for x in (train, dev, test)
    )
    return text_field, label_field, train_iter, dev_iter, test_iter, train, dev


def get_data_iterators_tacred(train_lm=False, map_cpu=False):
    text_field = tt.data.Field(lower=False)
    label_field = tt.data.LabelField()
    length_field = tt.data.Field(sequential=False, use_vocab=False)
    offset_field = tt.data.Field(sequential=False, use_vocab=False)
    pos_field = tt.data.Field()
    ner_field = tt.data.Field()
    subj_offset_field = tt.data.Field()
    obj_offset_field = tt.data.Field()

    path_format = "./.data/TACRED/data/json/%s.json"
    bert_tokenizer = None
    if args.use_bert_tokenizer:
        bert_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True, cache_dir="bert/cache"
        )
    train_examples, dev_examples, test_examples = (
        get_examples_tacred(path_format % ds, train_lm, bert_tokenizer=bert_tokenizer)
        for ds in ["train", "dev", "test"]
    )
    train, dev, test = (
        tt.data.Dataset(
            ex,
            [
                ("text", text_field),
                ("length", length_field),
                ("offset", offset_field),
                ("label", label_field),
                ("subj_offset", subj_offset_field),
                ("obj_offset", obj_offset_field),
                ("ner", ner_field),
                ("pos", pos_field),
            ],
        )
        for ex in [train_examples, dev_examples, test_examples]
    )

    vocab_path = (
        "vocab/vocab_tacred.pkl"
        if not args.use_bert_tokenizer
        else "vocab/vocab_tacred_bert.pkl"
    )
    if args.fix_test_vocab and not args.use_bert_tokenizer:
        vocab_path = "vocab/vocab_tacred_fix.pkl"

    c_postfix = ".tacred"
    if args.use_bert_tokenizer:
        c_postfix += ".bert"
    if args.fix_test_vocab:
        c_postfix += ".fix"
    handle_vocab(
        vocab_path,
        text_field,
        (train, dev, test),
        args.vector_cache + c_postfix,
        train_lm,
        max_size=100000,
    )
    handle_vocab(
        vocab_path,
        text_field,
        (train, dev, test),
        args.vector_cache + c_postfix,
        train_lm,
        max_size=100000,
    )
    handle_vocab(
        vocab_path + ".relation", label_field, (train, dev, test), "", False, None
    )
    handle_vocab(
        vocab_path + ".subj_offset",
        subj_offset_field,
        (train, dev, test),
        "",
        False,
        None,
    )
    handle_vocab(
        vocab_path + ".obj_offset",
        obj_offset_field,
        (train, dev, test),
        "",
        False,
        None,
    )
    handle_vocab(vocab_path + ".pos", pos_field, (train, dev, test), "", False, None)
    handle_vocab(vocab_path + ".ner", ner_field, (train, dev, test), "", False, None)

    train_iter, dev_iter, test_iter = (
        tt.data.BucketIterator(
            x,
            batch_size=args.batch_size,
            device=torch.device(args.gpu) if not map_cpu else "cpu",
        )
        for x in (train, dev, test)
    )
    return (
        text_field,
        label_field,
        subj_offset_field,
        obj_offset_field,
        pos_field,
        ner_field,
        train_iter,
        dev_iter,
        test_iter,
        train,
        dev,
    )
