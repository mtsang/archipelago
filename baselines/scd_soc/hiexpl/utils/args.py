from argparse import ArgumentParser
import os


def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
    avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def get_best_snapshot(dir):
    if os.path.isdir(dir):
        files = os.listdir(dir)
        for file in files:
            if file.startswith("best_"):
                return os.path.join(dir, file)
    return None


def get_args():
    parser = ArgumentParser(description="PyTorch/torchtext SST")

    # model parameters
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument(
        "--metrics", default="accuracy", choices=["accuracy", "tacred_f1"]
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--task", type=str, default="sst")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--d_embed", type=int, default=300)
    parser.add_argument("--d_proj", type=int, default=300)
    parser.add_argument("--d_hidden", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--dev_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--no-bidirectional", action="store_false", dest="birnn")
    parser.add_argument("--preserve-case", action="store_false", dest="lower")
    parser.add_argument("--no-projection", action="store_false", dest="projection")
    parser.add_argument("--fix_emb", action="store_true")
    parser.add_argument("--gpu", default=0)
    parser.add_argument("--save_path", type=str, default="results")
    parser.add_argument(
        "--vector_cache",
        type=str,
        default=os.path.join(os.getcwd(), ".vector_cache/input_vectors.pt"),
    )
    parser.add_argument("--word_vectors", type=str, default="glove.6B.300d")
    parser.add_argument("--resume_snapshot", type=str, default="")
    parser.add_argument("--word_dropout", action="store_true")

    parser.add_argument("--lm_d_embed", type=int, default=300)
    parser.add_argument("--lm_d_hidden", type=int, default=128)

    parser.add_argument("--method", nargs="?")
    parser.add_argument("--nb_method", default="ngram")
    parser.add_argument("--nb_range", type=int, default=3)
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--lm_dir", nargs="?", default="")
    parser.add_argument("--lm_path", nargs="?", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=10000000000)
    parser.add_argument("--sample_n", type=int, default=5)

    parser.add_argument("--explain_model", default="lstm")
    parser.add_argument("--demo", action="store_true")

    parser.add_argument("--dataset", default="dev")
    parser.add_argument("--use_bert_tokenizer", action="store_true")
    parser.add_argument("--no_subtrees", action="store_true")

    parser.add_argument("--use_bert_lm", action="store_true")
    parser.add_argument("--fix_test_vocab", action="store_true")

    parser.add_argument("--include_noise_labels", action="store_true")
    parser.add_argument("--filter_length_gt", type=int, default=-1)
    parser.add_argument("--add_itself", action="store_true")

    parser.add_argument("--mean_hidden", action="store_true")
    parser.add_argument("--agg", action="store_true")
    parser.add_argument("--class_score", action="store_true")

    parser.add_argument("--cd_pad", action="store_true")
    parser.add_argument("--eval_file", default="")
    parser.add_argument("-f", default="")

    args = parser.parse_args()

    try:
        args.gpu = int(args.gpu)
    except ValueError:
        args.gpu = "cpu"

    if os.path.isdir(args.resume_snapshot):
        args.resume_snapshot = get_best_snapshot(args.resume_snapshot)
    if os.path.isdir(args.lm_path):
        args.lm_path = get_best_snapshot(args.lm_path)
    return args


args = get_args()
