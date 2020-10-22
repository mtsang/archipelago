from torch import nn
import time
import glob
import sys

sys.path.append("../")

import torch.optim as O
from utils.reader import *
from utils.tacred_f1 import score as f1_score

from sklearn.metrics import accuracy_score


class BOWRegression(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.vocab_size = len(vocab.itos)
        self.weight = nn.Linear(self.vocab_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.gpu = config.gpu

    def forward(self, batch):
        text = batch.text.cpu().numpy()  # [T,B]
        bow = []
        for b in range(text.shape[1]):
            v = torch.zeros(self.vocab_size)
            for t in range(text.shape[0]):
                if text[t, b] != 1:  # [pad]
                    v[text[t, b]] = 1
            bow.append(v)
        bow = torch.stack(bow)  # [B, V]
        bow = bow.to(self.gpu)
        score = self.weight(bow)  # [B, 1]
        # loss = self.loss(score, batch.label.unsqueeze(-1))
        return self.sigmoid(score)

    def get_coefficient(self, idx):
        return self.weight.weight[0, idx].item()


class BOWRegressionMulti(nn.Module):
    def __init__(self, vocab, config, label_vocab):
        super().__init__()
        self.label_vocab = label_vocab
        self.vocab_size = len(vocab.itos)
        self.label_size = len(label_vocab.itos)
        self.weight = nn.Linear(self.vocab_size, self.label_size)
        self.softmax = nn.Softmax(-1)
        self.loss = nn.CrossEntropyLoss()

        self.gpu = config.gpu

    def forward(self, batch):
        text = batch.text.cpu().numpy()  # [T,B]
        bow = []
        for b in range(text.shape[1]):
            v = torch.zeros(self.vocab_size)
            for t in range(text.shape[0]):
                if text[t, b] != 1:  # [pad]
                    v[text[t, b]] = 1
            bow.append(v)
        bow = torch.stack(bow)  # [B, V]
        bow = bow.to(self.gpu)
        score = self.weight(bow)  # [B, C]
        # loss = self.loss(score, batch.label.unsqueeze(-1))
        return score

    def get_label_coefficient(self, class_idx_or_name, word_idx):
        if type(class_idx_or_name) is str:
            class_idx = self.label_vocab.stoi[class_idx_or_name]
        else:
            class_idx = class_idx_or_name
        return self.weight.weight[class_idx, word_idx].item()

    def get_margin_coefficient(self, class_idx_or_name, word_idx):
        if type(class_idx_or_name) is str:
            class_idx = self.label_vocab.stoi[class_idx_or_name]
        else:
            class_idx = class_idx_or_name
        weight_vec = self.weight.weight[:, word_idx]
        weight_vec_bak = weight_vec.clone()
        weight_vec_bak[class_idx] = -1000
        margin = weight_vec[class_idx] - weight_vec_bak.max()
        return margin.item()


def do_train():
    if args.task != "tacred":
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    opt = O.Adam(filter(lambda x: x.requires_grad, model.parameters()))

    iterations = 0
    start = time.time()
    best_dev_acc = -1
    best_dev_loss = 10000
    train_iter.repeat = False
    header = "  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy"
    dev_log_template = " ".join(
        "{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}".split(
            ","
        )
    )
    log_template = " ".join(
        "{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}".split(
            ","
        )
    )
    makedirs(args.save_path)
    print(header)

    all_break = False
    print(model)

    for epoch in range(args.epochs):
        if all_break:
            break
        train_iter.init_epoch()
        n_correct, n_total = 0, 0
        for batch_idx, batch in enumerate(train_iter):
            # switch model to training mode, clear gradient accumulators
            model.train()
            opt.zero_grad()

            iterations += 1
            print("epoch %d iter %d" % (epoch, batch_idx), end="\r")

            # forward pass
            answer = model(batch)

            # calculate loss of the network output with respect to training labels
            train_acc = 0
            if args.task != "tacred":
                batch.label = batch.label.float()
            loss = criterion(answer, batch.label.to(args.gpu))

            # backpropagate and update optimizer learning rate
            loss.backward()
            opt.step()

            # checkpoint model periodically
            if iterations % args.save_every == 0:
                snapshot_prefix = os.path.join(args.save_path, "snapshot")
                snapshot_path = (
                    snapshot_prefix
                    + "_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt".format(
                        train_acc, loss.item(), iterations
                    )
                )
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + "*"):
                    if f != snapshot_path:
                        os.remove(f)

            # evaluate performance on validation set periodically
            if iterations % args.dev_every == 0:

                # switch model to evaluation mode
                model.eval()
                dev_iter.init_epoch()
                avg_dev_loss = 0
                # calculate accuracy on validation set
                n_dev_correct, dev_loss = 0, 0
                truth_dev, pred_dev = [], []
                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                    answer = model(dev_batch)
                    dev_label = (
                        dev_batch.label
                        if not config.use_gpu
                        else dev_batch.label.cuda()
                    )
                    if args.task != "tacred":
                        pred = (answer > 0.5).long()
                    else:
                        pred = torch.max(answer, 1)[1]
                    if args.task != "tacred":
                        dev_label = dev_label.float()
                    dev_loss = criterion(answer, dev_label)
                    avg_dev_loss += dev_loss.item() * dev_label.size(0)
                    for l_i in range(dev_label.size(0)):
                        pred_dev.append(pred.view(-1)[l_i].item())
                        truth_dev.append(dev_label[l_i].item())
                if args.task in ["sst", "yelp"]:
                    dev_acc = 100.0 * accuracy_score(truth_dev, pred_dev)
                elif args.task == "tacred":
                    dev_acc = 100.0 * f1_score(truth_dev, pred_dev)[-1]
                else:
                    raise ValueError
                avg_dev_loss /= len(dev_set)
                print(
                    dev_log_template.format(
                        time.time() - start,
                        epoch,
                        iterations,
                        1 + batch_idx,
                        len(train_iter),
                        100.0 * (1 + batch_idx) / len(train_iter),
                        loss.item(),
                        avg_dev_loss,
                        train_acc,
                        dev_acc,
                    )
                )

                # update best valiation set accuracy
                if dev_acc > best_dev_acc:
                    # if avg_dev_loss < best_dev_loss:
                    best_dev_acc = dev_acc
                    best_dev_loss = avg_dev_loss
                    snapshot_prefix = os.path.join(args.save_path, "best_snapshot")
                    snapshot_path = (
                        snapshot_prefix
                        + "_devacc_{}_devloss_{}_iter_{}_model.pt".format(
                            dev_acc, dev_loss.item(), iterations
                        )
                    )

                    # save model, delete previous 'best_snapshot' files
                    torch.save(model, snapshot_path)
                    for f in glob.glob(snapshot_prefix + "*"):
                        if f != snapshot_path:
                            os.remove(f)

            elif iterations % args.log_every == 0:
                # print progress message
                print(
                    log_template.format(
                        time.time() - start,
                        epoch,
                        iterations,
                        1 + batch_idx,
                        len(train_iter),
                        100.0 * (1 + batch_idx) / len(train_iter),
                        loss.item(),
                        " " * 8,
                        n_correct / n_total * 100,
                        " " * 12,
                    )
                )


if __name__ == "__main__":

    if args.task == "sst":
        (
            inputs,
            labels,
            train_iter,
            dev_iter,
            test_iter,
            train_set,
            dev_set,
        ) = get_data_iterators_sst()
    elif args.task == "yelp":
        (
            inputs,
            labels,
            train_iter,
            dev_iter,
            test_iter,
            train_set,
            dev_set,
        ) = get_data_iterators_yelp()
    elif args.task == "tacred":
        (
            inputs,
            labels,
            subj_offset,
            obj_offset,
            pos,
            ner,
            train_iter,
            dev_iter,
            test_iter,
            train_set,
            dev_set,
        ) = get_data_iterators_tacred()
    else:
        raise ValueError("unknown task")

    config = args
    config.n_embed = len(inputs.vocab)
    config.d_out = len(labels.vocab)
    config.n_cells = config.n_layers
    config.use_gpu = args.gpu >= 0

    if args.task != "tacred":
        model = BOWRegression(inputs.vocab, config)
    else:
        model = BOWRegressionMulti(inputs.vocab, config, labels.vocab)
    if config.use_gpu:
        model = model.cuda()

    # if not args.bow_snapshot:
    do_train()
