import torch
import numpy as np
from tqdm import tqdm
import copy
from utils.general_utils import *
from utils.text_utils import *
from utils.graph_utils import *
from utils.dna_utils import *
from utils.lime.lime_text import *


def generate_binary_perturbations(
    num_feat, num_samples=100, init_on=True, perturbed_features=None
):
    if perturbed_features == None:
        perturbed_features = {"indices": np.array(range(num_feat))}
    num_perturb = len(perturbed_features["indices"])

    samples_binary = np.ones((num_samples, num_feat), dtype=np.int8)
    perturb_binary = np.ones((num_samples, num_perturb), dtype=np.int8)
    num_flips = np.random.randint(1, num_perturb + 1, num_samples)

    for r in range(num_samples):
        if not (init_on and r == 0):
            num_flip = num_flips[r]
            perturb_binary[r, 0:num_flip] = np.zeros(num_flip, dtype=np.int8)
            np.random.shuffle(perturb_binary[r])
        samples_binary[r, perturbed_features["indices"]] = perturb_binary[r]

    return samples_binary


def generate_perturbation_dataset_autoint(
    data_inst,
    model,
    dense_feat_indices,
    sparse_feat_indices,
    num_samples=6000,
    seed=None,
    **kwargs
):
    if seed is not None:
        set_seed(seed)

    def inv_sigmoid(y):
        return np.log(y / (1 - y))

    num_feats = len(dense_feat_indices) + len(sparse_feat_indices)
    samples_binary = generate_binary_perturbations(num_feats, num_samples, True)

    means_arr = np.array([data_inst["means"][i] for i in dense_feat_indices])

    perturb_Xv = []
    perturb_Xi = []
    for i in range(num_samples):
        raw_dense = data_inst["Xv"][dense_feat_indices]
        raw_sparse = data_inst["Xv"][sparse_feat_indices]
        binary_dense = samples_binary[i, dense_feat_indices]
        binary_sparse = samples_binary[i, sparse_feat_indices]
        perturb_raw_dense = raw_dense + binary_dense + means_arr * (1 - binary_dense)
        perturb_raw_sparse = binary_sparse * raw_sparse

        perturb_raw = np.zeros(num_feats)
        perturb_raw[dense_feat_indices] = perturb_raw_dense
        perturb_raw[sparse_feat_indices] = perturb_raw_sparse

        # perturb_raw = np.concatenate([perturb_raw_dense, perturb_raw_sparse])
        perturb_Xv.append(perturb_raw)
        perturb_Xi.append(data_inst["Xi"])
    perturb_Xv = np.stack(perturb_Xv)
    perturb_Xi = np.stack(perturb_Xi)

    samples_labels = inv_sigmoid(model.predict(perturb_Xi, perturb_Xv))

    Xs, Ys = proprocess_data(samples_binary.astype(np.int64), samples_labels, **kwargs)
    return Xs, Ys


def generate_perturbation_dataset_image(
    data_inst,
    model,
    class_idx,
    device,
    num_samples=6000,
    batch_size=100,
    seed=None,
    **kwargs
):
    # Based on LIME image: https://github.com/marcotcr/lime/blob/master/lime/lime_image.py

    if seed is not None:
        set_seed(seed)

    image = data_inst["orig"]
    segments = data_inst["segments"]
    num_feats = len(np.unique(segments))

    samples_binary = generate_binary_perturbations(num_feats, num_samples, True)

    image_means = image.copy()
    for i in np.unique(segments):
        image_means[segments == i] = (
            np.mean(image[segments == i][:, 0]),
            np.mean(image[segments == i][:, 1]),
            np.mean(image[segments == i][:, 2]),
        )

    n_batches = int(np.ceil(num_samples / batch_size))

    samples_labels = []
    for i in tqdm(range(n_batches)):

        samples_binary_batch = samples_binary[i * batch_size : (i + 1) * batch_size]

        perturbed_imgs = []
        for sample_binary in samples_binary_batch:
            temp = copy.deepcopy(image)
            zeros = np.where(sample_binary == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = image_means[mask]

            perturbed_imgs.append(temp)

        torch_img = (
            torch.FloatTensor(np.array(perturbed_imgs)).to(device).permute(0, 3, 1, 2)
        )
        preds = model(torch_img).data.cpu().numpy()
        samples_labels.extend(preds)

    samples_labels = np.stack(samples_labels)

    Xs, Ys = proprocess_data(samples_binary, samples_labels[:, class_idx], **kwargs)

    return Xs, Ys


def generate_perturbation_dataset_text(
    data_inst,
    model,
    class_idx,
    device,
    num_samples=6000,
    batch_size=100,
    seed=None,
    model_id=None,
    **kwargs
):
    # Based on LIME image: https://github.com/marcotcr/lime/blob/master/lime/lime_text.py

    if seed is not None:
        set_seed(seed)

    text = data_inst["orig"]

    indexed_string = IndexedString(text, bow=False)
    data_inst["domain_mapper"] = TextDomainMapper(indexed_string)

    num_feats = indexed_string.num_words()

    samples_binary = generate_binary_perturbations(num_feats, num_samples, True)

    n_batches = int(np.ceil(num_samples / batch_size))

    samples_labels = []
    for i in tqdm(range(n_batches)):

        samples_binary_batch = samples_binary[i * batch_size : (i + 1) * batch_size]

        perturbed_text = []
        for sample_binary in samples_binary_batch:

            indices2invert = np.argwhere(sample_binary == 0).squeeze()
            inv = indexed_string.inverse_removing(indices2invert)

            if model_id == "bert":
                ex = inv
            else:
                ex = data.Example.fromlist(
                    [inv], fields=[("text", data_inst["vectorizer"])]
                )
            perturbed_text.append(ex)

        if model_id == "bert":
            preds = model(perturbed_text)
        else:
            dset = data.Dataset(
                perturbed_text, fields=[("text", data_inst["vectorizer"])]
            )
            test_samples = data.Batch(data=perturbed_text, dataset=dset, device=device)
            preds = model(test_samples).data.cpu().numpy()

        samples_labels.append(preds)

    samples_labels = np.concatenate(samples_labels)

    Xs, Ys = proprocess_data(samples_binary, samples_labels[:, class_idx], **kwargs)

    return Xs, Ys


def generate_perturbation_dataset_graph(
    data_inst,
    model,
    target_idx,
    n_hops,
    device,
    num_samples=6000,
    batch_size=500,
    seed=None,
    **kwargs
):
    def get_output(x, da):
        return model(x, da)[test_idxs].detach().cpu()

    if seed is not None:
        set_seed(seed)

    node_feats = data_inst["nodes"]
    adj_mat = data_inst["edges"]
    test_idxs = data_inst["test_idxs"]

    da_mat = convert_adj_to_da(adj_mat)

    # Collect all nodes within a k-hop neighborhood of the target test index
    adj_cum = copy.deepcopy(adj_mat)
    for i in range(n_hops - 1):
        adj_cum = torch.matmul(adj_cum, adj_mat)

    sum_v = 0
    counter = 0
    locality_dict = dict()
    locality_dict_rev = dict()
    for i, v in enumerate(adj_cum[target_idx]):
        if v != 0:
            sum_v += v
            locality_dict[i] = counter
            locality_dict_rev[counter] = i
            counter += 1
    local_num_nodes = len(locality_dict)

    data_inst["local_idx_map"] = locality_dict_rev

    samples_binary = generate_binary_perturbations(local_num_nodes, num_samples, True)

    # Get the features associated binary samples
    data_new = []
    for i in range(node_feats.shape[0]):
        if i in locality_dict:
            data_new.append(samples_binary[:, locality_dict[i]])
        else:
            data_new.append(np.zeros(num_samples))
    data_new = np.array(data_new).transpose()

    # Get the test predictions associated binary samples
    results = []
    for d in tqdm(data_new):
        mask = torch.FloatTensor(d).view(-1, 1).expand(node_feats.size())
        masked_features = node_feats * mask.to(device)
        output = get_output(masked_features, da_mat).numpy()
        results.append(output)
    results = np.array(results)

    y_idx = test_idxs.index(target_idx)
    classifications = get_output(node_feats, da_mat).max(1)[1]

    samples_labels = results[:, y_idx, classifications[y_idx]]

    #     samples_labels = []
    #     for ci, c in enumerate(classifications):
    #         samples_labels.append(results[:, ci, c])

    Xs, Ys = proprocess_data(samples_binary, samples_labels, **kwargs)

    return Xs, Ys


def generate_perturbation_dataset_dna(
    data_inst, model, device, num_samples=6000, batch_size=100, seed=None, **kwargs
):

    if seed is not None:
        set_seed(seed)

    seq = data_inst["orig"]
    vectorizer = data_inst["vectorizer"]

    indexed_seq = IndexedNucleotides(seq)

    num_feats = indexed_seq.num_nucleotides()

    samples_binary = generate_binary_perturbations(num_feats, num_samples, True)

    n_batches = int(np.ceil(num_samples / batch_size))

    samples_labels = []
    for i in tqdm(range(n_batches)):

        samples_binary_batch = samples_binary[i * batch_size : (i + 1) * batch_size]

        perturbed_seqs = []
        for sample_binary in samples_binary_batch:

            indices2invert = np.argwhere(sample_binary == 0).squeeze()
            inv = indexed_seq.perturb_nucleotide(indices2invert)
            ex = vectorizer(inv)
            perturbed_seqs.append(ex)

        test_samples = torch.FloatTensor(perturbed_seqs).permute(0, 2, 1).to(device)
        preds = model(test_samples).data.cpu().numpy()

        samples_labels.append(preds)

    samples_labels = np.concatenate(samples_labels).squeeze()

    Xs, Ys = proprocess_data(samples_binary, samples_labels, **kwargs)

    return Xs, Ys
