import time
import glob

import sys

sys.path.append("../")

import torch.optim as O

from lm_arch import LSTMLanguageModel
from utils.reader import *


import random

random.seed(0)

args = get_args()
try:
    torch.cuda.set_device(args.gpu)
except AttributeError:
    pass


def do_train(model):
    opt = O.Adam(filter(lambda x: x.requires_grad, model.parameters()))

    iterations = 0
    start = time.time()
    best_dev_nll = 1e10
    train_iter.repeat = False
    header = "  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss"
    dev_log_template = " ".join(
        "{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:>8.6f},{:>8.6f}".split(
            ","
        )
    )
    log_template = " ".join(
        "{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{}".split(",")
    )
    makedirs(args.save_path)
    print(header)

    all_break = False
    print(model)

    for epoch in range(args.epochs):
        if all_break:
            break
        train_iter.init_epoch()
        train_loss = 0
        for batch_idx, batch in enumerate(train_iter):
            # switch model to training mode, clear gradient accumulators
            model.train()
            opt.zero_grad()

            iterations += 1
            print(("epoch %d iter %d" + " " * 10) % (epoch, batch_idx), end="\r")

            # forward pass
            fw_loss, bw_loss = model(batch)

            loss = fw_loss + bw_loss
            # backpropagate and update optimizer learning rate
            loss.backward()
            opt.step()

            train_loss += loss.item()

            # checkpoint model periodically
            if iterations % args.save_every == 0:
                snapshot_prefix = os.path.join(args.save_path, "snapshot")
                snapshot_path = snapshot_prefix + "loss_{:.6f}_iter_{}_model.pt".format(
                    loss.item(), iterations
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

                # calculate accuracy on validation set

                cnt, dev_loss = 0, 0
                dev_fw_loss, dev_bw_loss = 0, 0
                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                    fw_loss, bw_loss = model(dev_batch)
                    loss = fw_loss + bw_loss
                    cnt += 1
                    dev_loss += loss.item()
                    dev_fw_loss += fw_loss.item()
                    dev_bw_loss += bw_loss.item()
                dev_loss /= cnt
                dev_fw_loss /= cnt
                dev_bw_loss /= cnt
                print(
                    dev_log_template.format(
                        time.time() - start,
                        epoch,
                        iterations,
                        1 + batch_idx,
                        len(train_iter),
                        100.0 * (1 + batch_idx) / len(train_iter),
                        train_loss / (batch_idx + 1),
                        dev_loss,
                        dev_fw_loss,
                        dev_bw_loss,
                    )
                )

                # update best valiation set accuracy
                if dev_loss < best_dev_nll:
                    best_dev_nll = dev_loss
                    snapshot_prefix = os.path.join(args.save_path, "best_snapshot")
                    snapshot_path = (
                        snapshot_prefix
                        + "_devloss_{}_iter_{}_model.pt".format(dev_loss, iterations)
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
                    )
                )


if __name__ == "__main__":
    if args.task == "sst":
        (
            inputs,
            lengths,
            train_iter,
            dev_iter,
            test_iter,
            train_set,
            dev_set,
        ) = get_data_iterators_sst_flatten(train_lm=True)
    elif args.task == "yelp":
        (
            inputs,
            labels,
            train_iter,
            dev_iter,
            test_iter,
            train_set,
            dev_set,
        ) = get_data_iterators_yelp(train_lm=True)
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
        ) = get_data_iterators_tacred(train_lm=True)
    else:
        raise ValueError("unknown task")

    config = args
    config.n_embed = len(inputs.vocab)
    config.n_cells = config.n_layers
    config.use_gpu = args.gpu >= 0

    model = LSTMLanguageModel(config, inputs.vocab)
    if args.word_vectors:
        model.encoder.embedding.weight.data = inputs.vocab.vectors
        if args.fix_emb:
            model.encoder.embedding.weight.requires_grad = False
    if config.use_gpu:
        model = model.cuda()

    do_train(model)
