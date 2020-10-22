## AutoInt's official training code. Modifications are only made to accomodate cross feature

import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from time import time
from model import AutoInt
import argparse
import os

from os.path import join


def str2list(v):
    v = v.split(",")
    v = [int(_.strip("[]")) for _ in v]

    return v


def str2list2(v):
    v = v.split(",")
    v = [float(_.strip("[]")) for _ in v]

    return v


def str2bool(v):
    if v.lower() in ["yes", "true", "t", "y", "1"]:
        return True
    elif v.lower() in ["no", "false", "f", "n", "0"]:
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=3, help="#blocks")
    parser.add_argument(
        "--block_shape",
        type=str2list,
        default=[64, 64, 64],
        help="output shape of each block",
    )
    parser.add_argument("--heads", type=int, default=2, help="#heads")
    parser.add_argument("--embedding_size", type=int, default=16)
    parser.add_argument("--dropout_keep_prob", type=str2list2, default=[1, 1, 1])
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--learning_rate_wide", type=float, default=0.001)
    parser.add_argument("--optimizer_type", type=str, default="adam")
    parser.add_argument("--l2_reg", type=float, default=0.0)
    parser.add_argument("--random_seed", type=int, default=2018)
    parser.add_argument("--save_path", type=str, default="./model/")
    parser.add_argument("--field_size", type=int, default=0, help="dummy variable")
    parser.add_argument("--loss_type", type=str, default="logloss")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--run_times", type=int, default=5, help="run multiple times to eliminate error"
    )
    parser.add_argument("--is_save", type=str2bool, default=True)
    parser.add_argument(
        "--greater_is_better", type=str2bool, default=False, help="early stop criterion"
    )
    parser.add_argument(
        "--has_residual", type=str2bool, default=True, help="add residual or not"
    )
    parser.add_argument("--has_wide", type=str2bool, default=False)
    parser.add_argument(
        "--deep_layers",
        type=str2list,
        default=[400, 400],
        help="config for dnn in joint train",
    )
    parser.add_argument("--batch_norm", type=int, default=0)
    parser.add_argument("--batch_norm_decay", type=float, default=0.995)
    parser.add_argument("--data", type=str, help="data name")
    parser.add_argument(
        "--data_path", type=str, default="./", help="root path for all the data"
    )
    parser.add_argument("--gpu", type=int, help="which gpu")
    parser.add_argument("--exp", type=str, help="experiment", default="cross")
    parser.add_argument("--cross_exp", type=str, help="cross exp", default="cross1")

    return parser.parse_args()


def include_cross_features(args, Xi, Xv, j):
    if args.exp == "cross":
        path = join(args.data_path, args.data, "part" + str(j), args.cross_exp)

        Xi_cross = np.load(join(path, "i_cross.npy"))
        Xv_cross = np.load(join(path, "x_cross.npy"))
        Xi = np.concatenate([Xi, Xi_cross], axis=1)
        Xv = np.concatenate([Xv, Xv_cross], axis=1)
    return Xi, Xv


def _run_(args, file_name, run_cnt):
    # path_prefix = '../Dataprocess/' + args.data
    path_prefix = os.path.join(args.data_path, args.data)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.exp == "cross":
        feature_size = np.load(join(path_prefix, args.cross_exp, "feature_size.npy"))[0]
    else:
        feature_size = np.load(path_prefix + "/feature_size.npy")[0]

    # variables = tf.contrib.framework.get_variables_to_restore()
    # print(variables)
    # return

    Xi_valid = np.load(path_prefix + "/part2/" + file_name[0])
    Xv_valid = np.load(path_prefix + "/part2/" + file_name[1])
    y_valid = np.load(path_prefix + "/part2/" + file_name[2])

    Xi_valid, Xv_valid = include_cross_features(args, Xi_valid, Xv_valid, 2)

    args.field_size = Xi_valid.shape[1]

    # test: file1, valid: file2, train: file3-10
    model = AutoInt(args=args, feature_size=feature_size, run_cnt=run_cnt)

    is_continue = True
    for k in range(model.epoch):
        if not is_continue:
            print("early stopping at epoch %d" % (k + 1))
            break
        file_count = 0
        time_epoch = 0
        for j in range(3, 11):
            if not is_continue:
                print("early stopping at epoch %d file %d" % (k + 1, j))
                break
            file_count += 1
            Xi_train = np.load(path_prefix + "/part" + str(j) + "/" + file_name[0])
            Xv_train = np.load(path_prefix + "/part" + str(j) + "/" + file_name[1])
            y_train = np.load(path_prefix + "/part" + str(j) + "/" + file_name[2])

            Xi_train, Xv_train = include_cross_features(args, Xi_train, Xv_train, j)

            print("epoch %d, file %d" % (k + 1, j))
            t1 = time()
            is_continue = model.fit_once(
                Xi_train,
                Xv_train,
                y_train,
                k + 1,
                file_count,
                Xi_valid,
                Xv_valid,
                y_valid,
                early_stopping=True,
            )
            time_epoch += time() - t1

        print("epoch %d, time %d" % (k + 1, time_epoch))

    print("start testing!...")
    Xi_test = np.load(path_prefix + "/part1/" + file_name[0])
    Xv_test = np.load(path_prefix + "/part1/" + file_name[1])
    y_test = np.load(path_prefix + "/part1/" + file_name[2])

    Xi_test, Xv_test = include_cross_features(args, Xi_test, Xv_test, 1)

    model.restore()

    test_result, test_loss = model.evaluate(Xi_test, Xv_test, y_test)
    print("test-result = %.4lf, test-logloss = %.4lf" % (test_result, test_loss))
    return test_result, test_loss


if __name__ == "__main__":
    args = parse_args()
    print(args.__dict__)
    print("**************")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    data_path = args.data.split("/")
    if any([data_path[-1].startswith(d) for d in ["avazu"]]):
        file_name = ["train_i.npy", "train_x.npy", "train_y.npy"]
    elif any([data_path[-1].startswith(d) for d in ["criteo"]]):
        file_name = ["train_i.npy", "train_x2.npy", "train_y.npy"]
    else:
        raise ValueError("invalid data arg")
    test_auc = []
    test_log = []

    print("run time : %d" % args.run_times)
    for i in range(1, args.run_times + 1):
        test_result, test_loss = _run_(args, file_name, i)
        test_auc.append(test_result)
        test_log.append(test_loss)
    print("test_auc", test_auc)
    print("test_log_loss", test_log)
    print("avg_auc", sum(test_auc) / len(test_auc))
    print("avg_log_loss", sum(test_log) / len(test_log))
