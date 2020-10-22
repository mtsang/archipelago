from autoint.model import AutoInt
from application_utils.common_utils import get_efficient_mask_indices
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pickle


class AutoIntWrapper:
    def __init__(self, model, Xi_inst, inv_sigmoid=True):
        self.model = model
        self.Xi_inst = Xi_inst
        self.use_inv_sigmoid = inv_sigmoid

    def inv_sigmoid(self, y):
        return np.log(y / (1 - y))

    def __call__(self, Xv):
        Xi = np.repeat(self.Xi_inst, Xv.shape[0], axis=0)
        pred = self.model.predict(Xi, Xv)
        if self.use_inv_sigmoid:
            pred = self.inv_sigmoid(pred)
        return np.expand_dims(pred, 1)


class IdXformer:
    def __init__(self, input_ids, baseline_ids):
        self.input = input_ids.flatten()
        self.baseline = baseline_ids.flatten()
        self.num_features = len(self.input)

    def efficient_xform(self, inst):
        mask_indices, base, change = get_efficient_mask_indices(
            inst, self.baseline, self.input
        )
        for i in mask_indices:
            base[i] = change[i]
        return base

    def __call__(self, inst):
        id_list = self.efficient_xform(inst)
        return id_list


def evaluate(model, data, batch_size=1000):
    num_samples = data["Xi"].shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))

    preds = []
    for i in tqdm(range(num_batches)):
        Xi_batch = data["Xi"][i * batch_size : (i + 1) * batch_size]
        Xv_batch = data["Xv"][i * batch_size : (i + 1) * batch_size]
        pred_batch = model.predict(Xi_batch, Xv_batch)
        preds.append(pred_batch)

    y_pred = np.concatenate(preds)
    y_gt = data["y"][:num_samples]

    return roc_auc_score(y_gt, y_pred)


def get_example(data, index):
    Xv_inst = data["Xv"][index : index + 1]
    Xi_inst = data["Xi"][index : index + 1]
    return Xv_inst, Xi_inst


def get_autoint_and_data(
    dataset=None,
    data_path=None,
    save_path=None,
    feature_size=1544489,
):
    args = parse_args(dataset, data_path, save_path)

    run_cnt = 0
    model = AutoInt(args=args, feature_size=feature_size, run_cnt=run_cnt)
    model.restore(args.save_path)

    with open(data_path, "rb") as handle:
        data_batch = pickle.load(handle)
    return model, data_batch


def get_avazu_dict():
    avazu_dict = {
        0: "id: ad identifier",
        1: "hour",
        2: "C1",
        3: "banner_pos",
        4: "site_id",
        5: "site_domain",
        6: "site_category",
        7: "app_id",
        8: "app_domain",
        9: "app_category",
        10: "device_id",
        11: "device_ip",
        12: "device_model",
        13: "device_type",
        14: "device_conn_type",
    }
    for i in range(15, 23):
        avazu_dict[i] = "C" + str(i - 1)
    return avazu_dict


def parse_args(dataset, data_path, save_path):
    dataset = dataset.lower()
    if "avazu" in dataset:
        field_size = 23
    elif "criteo" in dataset:
        field_size = 39
    else:
        raise ValueError("Invalid dataset")

    return get_args(save_path, field_size, dataset, data_path)


def get_data_info(args):
    data = args.data.split("/")[-1].lower()
    if any([data.startswith(d) for d in ["avazu"]]):
        file_name = ["train_i.npy", "train_x.npy", "train_y.npy"]
    elif any([data.startswith(d) for d in ["criteo"]]):
        file_name = ["train_i.npy", "train_x2.npy", "train_y.npy"]
    else:
        raise ValueError("invalid data arg")

    path_prefix = os.path.join(args.data_path, args.data)
    return file_name, path_prefix


class get_args:
    # the original parameter configuration of AutoInt
    blocks = 3
    block_shape = [64, 64, 64]
    heads = 2
    embedding_size = 16
    dropout_keep_prob = [1, 1, 1]
    epoch = 3
    batch_size = 1024
    learning_rate = 0.001
    learning_rate_wide = 0.001
    optimizer_type = "adam"
    l2_reg = 0.0
    random_seed = 2018  # used in the official autoint code
    loss_type = "logloss"
    verbose = 1
    run_times = 1
    is_save = False
    greater_is_better = False
    has_residual = True
    has_wide = False
    deep_layers = [400, 400]
    batch_norm = 0
    batch_norm_decay = 0.995

    def __init__(self, save_path, field_size, dataset, data_path):
        self.save_path = save_path
        self.field_size = field_size
        self.data = dataset
        self.data_path = data_path
