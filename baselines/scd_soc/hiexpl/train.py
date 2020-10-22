import time
import glob

import torch.optim as O

from nns.model import *
from utils.reader import *
from utils.tacred_f1 import score as tacred_f1_score

import random

random.seed(0)

args = get_args()
try:
    torch.cuda.set_device(args.gpu)
except AttributeError:
    pass


def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def word_dropout(train_batch):
    for b in range(train_batch.length.size(0)):
        for t in range(train_batch.length[b].item()):
            if random.random() < 0.04:
                train_batch.text[t, b] = 0  # unk


def do_train():
    criterion = nn.CrossEntropyLoss()
    if args.optim == "adam":
        opt = O.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            weight_decay=1e-6,
            lr=args.lr,
        )
    else:
        opt = O.SGD(
            filter(lambda x: x.requires_grad, model.parameters()),
            weight_decay=1e-6,
            lr=args.lr,
        )
    iterations = 0
    start = time.time()
    best_dev_acc = -1
    prev_dev_acc = -1
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
            if args.word_dropout:
                word_dropout(batch)
            iterations += 1
            print("epoch %d iter %d" % (epoch, batch_idx), end="\r")

            # forward pass
            answer = model(batch)

            # calculate accuracy of predictions in the current batch
            label = batch.label if not config.use_gpu else batch.label.cuda()
            n_correct += (
                torch.max(answer, 1)[1].view(label.size()).data == label.data
            ).sum()
            n_total += batch.batch_size
            train_acc = 100.0 * n_correct / n_total

            # calculate loss of the network output with respect to training labels
            loss = criterion(answer, label)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            # backpropagate and update optimizer learning rate
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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
                torch.save(model.state_dict(), snapshot_path)
                for f in glob.glob(snapshot_prefix + "*"):
                    if f != snapshot_path:
                        os.remove(f)

            # evaluate performance on validation set periodically
            if iterations % args.dev_every == 0:

                # switch model to evaluation mode
                model.eval()
                dev_iter.init_epoch()

                # calculate accuracy on validation set
                n_dev_correct, dev_loss_avg = 0, 0
                truth_dev, pred_dev = [], []
                with torch.no_grad():
                    for dev_batch_idx, dev_batch in enumerate(dev_iter):
                        answer = model(dev_batch)
                        dev_label = (
                            dev_batch.label
                            if not config.use_gpu
                            else dev_batch.label.cuda()
                        )
                        pred = torch.max(answer, 1)[1]
                        n_dev_correct += (
                            pred.view(dev_label.size()).data == dev_label.data
                        ).sum()
                        dev_loss = criterion(answer, dev_label)
                        for l_i in range(dev_label.size(0)):
                            pred_dev.append(pred.view(-1)[l_i].item())
                            truth_dev.append(dev_label[l_i].item())
                        dev_loss_avg += dev_loss.item()
                if args.metrics == "tacred_f1":
                    dev_acc = 100.0 * tacred_f1_score(truth_dev, pred_dev)[-1]
                else:
                    dev_acc = 100.0 * n_dev_correct / len(dev_set)
                print(
                    dev_log_template.format(
                        time.time() - start,
                        epoch,
                        iterations,
                        1 + batch_idx,
                        len(train_iter),
                        100.0 * (1 + batch_idx) / len(train_iter),
                        loss.item(),
                        dev_loss_avg / len(dev_iter),
                        train_acc,
                        dev_acc,
                    )
                )

                # update best valiation set accuracy
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    snapshot_prefix = os.path.join(args.save_path, "best_snapshot")
                    snapshot_path = (
                        snapshot_prefix
                        + "_devacc_{}_devloss_{}_iter_{}_model.pt".format(
                            dev_acc, dev_loss.item(), iterations
                        )
                    )

                    # save model, delete previous 'best_snapshot' files
                    torch.save(model.state_dict(), snapshot_path)
                    for f in glob.glob(snapshot_prefix + "*"):
                        if f != snapshot_path:
                            os.remove(f)

                # if iterations > 15000 and dev_acc < prev_dev_acc:
                #    args.lr *= 0.9
                #    change_lr(opt, args.lr)
                #    prev_dev_acc = dev_acc
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


def do_test():
    iterations = 0
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(test_iter):
        # switch model to training mode, clear gradient accumulators
        model.train()
        iterations += 1

        # forward pass
        answer = model(batch)

        # calculate accuracy of predictions in the current batch
        label = batch.label if not config.use_gpu else batch.label.cuda()
        n_correct += (
            torch.max(answer, 1)[1].view(label.size()).data == label.data
        ).sum()
        n_total += batch.batch_size
        acc = 100.0 * n_correct / n_total
        if batch_idx % 100 == 0:
            print(
                "evaluating %d\tacc: %f, %d, %d" % (batch_idx, acc, n_correct, n_total)
            )
    print("Acc: %f" % (n_correct / n_total))


if __name__ == "__main__":
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    config = args
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
        ) = get_data_iterators_yelp(map_cpu=False)
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
        config.subj_offset_size = len(subj_offset.vocab)
        config.obj_offset_size = len(obj_offset.vocab)
        config.pos_size = len(pos.vocab)
        config.ner_size = len(ner.vocab)
        config.offset_emb_dim = 30

    else:
        raise ValueError("unknown task")

    config.n_embed = len(inputs.vocab)
    config.d_out = len(labels.vocab) if hasattr(labels, "vocab") else 2
    config.n_cells = config.n_layers
    config.use_gpu = args.gpu >= 0

    if args.task == "sst":
        model = LSTMSentiment(config)
    elif args.task == "yelp":
        model = LSTMMeanSentiment(config, match_length=True)
    elif args.task == "tacred":
        model = LSTMMeanRE(config, match_length=True)
        model.init_weights(inputs.vocab)
    if args.word_vectors:
        model.embed.weight.data.copy_(inputs.vocab.vectors)
    if config.use_gpu:
        model = model.to(args.gpu)

    do_train()
