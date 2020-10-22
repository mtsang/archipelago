from utils.args import *
import torch.nn.functional as F
import torch
from .cd_func import is_in_intervals, sigmoid, tanh

args = get_args()


def CD_gpu(batch, model, intervals, hist_states, gpu):
    # if task is tacred, then the word vecs is more complicated
    if not args.task == "tacred":
        word_vecs = model.embed(batch.text)[:, 0]
        lstm_module = model.lstm
    else:
        token_vec = model.embed(batch.text)
        pos_vec = model.pos_embed(batch.pos)
        ner_vec = model.ner_embed(batch.ner)
        word_vecs = torch.cat([token_vec, pos_vec, ner_vec], -1)[:, 0]
        lstm_module = model.lstm.rnn

    hidden_dim = model.hidden_dim
    T = word_vecs.size(0)
    relevant, irrelevant, relevant_h, irrelevant_h = (
        torch.zeros(T, hidden_dim).to(gpu),
        torch.zeros(T, hidden_dim).to(gpu),
        torch.zeros(T, hidden_dim).to(gpu),
        torch.zeros(T, hidden_dim).to(gpu),
    )
    W_ii, W_if, W_ig, W_io = torch.split(
        lstm_module.weight_ih_l0, lstm_module.hidden_size
    )
    W_hi, W_hf, W_hg, W_ho = torch.split(
        lstm_module.weight_hh_l0, lstm_module.hidden_size
    )
    b_i, b_f, b_g, b_o = torch.split(
        lstm_module.bias_ih_l0 + lstm_module.bias_hh_l0, lstm_module.hidden_size
    )

    i_states, g_states, f_states, c_states = [], [], [None], []
    o_states = []

    for i in range(T):
        bias_in_rel = is_in_intervals(i, intervals)

        if i > 0:
            prev_rel_h = relevant_h[i - 1]
            prev_irrel_h = irrelevant_h[i - 1]
        else:
            prev_rel_h = torch.zeros(hidden_dim).to(gpu)
            prev_irrel_h = torch.zeros(hidden_dim).to(gpu)
        rel_i = torch.matmul(W_hi, prev_rel_h)
        rel_g = torch.matmul(W_hg, prev_rel_h)
        rel_f = torch.matmul(W_hf, prev_rel_h)
        rel_o = torch.matmul(W_ho, prev_rel_h)
        irrel_i = torch.matmul(W_hi, prev_irrel_h)
        irrel_g = torch.matmul(W_hg, prev_irrel_h)
        irrel_f = torch.matmul(W_hf, prev_irrel_h)
        irrel_o = torch.matmul(W_ho, prev_irrel_h)
        if is_in_intervals(i, intervals):
            rel_i = rel_i + torch.matmul(W_ii, word_vecs[i])
            rel_g = rel_g + torch.matmul(W_ig, word_vecs[i])
            rel_f = rel_f + torch.matmul(W_if, word_vecs[i])
            rel_o = rel_o + torch.matmul(W_io, word_vecs[i])
        else:
            irrel_i = irrel_i + torch.matmul(W_ii, word_vecs[i])
            irrel_g = irrel_g + torch.matmul(W_ig, word_vecs[i])
            irrel_f = irrel_f + torch.matmul(W_if, word_vecs[i])
            irrel_o = irrel_o + torch.matmul(W_io, word_vecs[i])
        (
            rel_contrib_i,
            irrel_contrib_i,
            bias_contrib_i,
        ) = decomp_activation_three_with_states(
            rel_i, irrel_i, b_i, F.sigmoid, hist_states, "i", i, bias_in_rel
        )
        (
            rel_contrib_g,
            irrel_contrib_g,
            bias_contrib_g,
        ) = decomp_activation_three_with_states(
            rel_g, irrel_g, b_g, F.tanh, hist_states, "g", i, bias_in_rel
        )

        i_states.append(sum([rel_i, irrel_i, b_i]))
        g_states.append(sum([rel_g, irrel_g, b_g]))

        relevant[i], irrelevant[i] = mult_terms(
            rel_contrib_g,
            irrel_contrib_g,
            bias_contrib_g,
            rel_contrib_i,
            irrel_contrib_i,
            bias_contrib_i,
            hist_states,
            "act_g",
            "act_i",
            i,
            bias_in_rel,
        )

        if i > 0:
            (
                rel_contrib_f,
                irrel_contrib_f,
                bias_contrib_f,
            ) = decomp_activation_three_with_states(
                rel_f, irrel_f, b_f, F.sigmoid, hist_states, "f", i, bias_in_rel
            )

            rel_plus, irrel_plus = mult_terms(
                rel_contrib_f,
                irrel_contrib_f,
                bias_contrib_f,
                relevant[i - 1],
                irrelevant[i - 1],
                0,
                hist_states,
                "act_f",
                "temp_c",
                i,
                bias_in_rel,
            )
            relevant[i] += rel_plus
            irrelevant[i] += irrel_plus

            f_states.append(sum([rel_f, irrel_f, b_f]))

        o = sigmoid(
            torch.matmul(W_io, word_vecs[i])
            + torch.matmul(W_ho, prev_rel_h + prev_irrel_h)
            + b_o
        )
        (
            rel_contrib_o,
            irrel_contrib_o,
            bias_contrib_o,
        ) = decomp_activation_three_with_states(
            rel_o, irrel_o, b_o, F.sigmoid, hist_states, "o", i, bias_in_rel
        )
        new_rel_h, new_irrel_h = decomp_activation_two_with_states(
            relevant[i], irrelevant[i], F.tanh, hist_states, "c", i
        )
        c_states.append(sum([relevant[i], irrelevant[i]]))
        o_states.append(o)
        relevant_h[i], irrelevant_h[i] = mult_terms(
            rel_contrib_o,
            irrel_contrib_o,
            bias_contrib_o,
            new_rel_h,
            new_irrel_h,
            0,
            hist_states,
            "o",
            "tanhc",
            i,
            bias_in_rel,
        )

    W_out = model.hidden_to_label.weight.data

    if hasattr(model, "drop"):
        relevant_h[T - 1], irrelevant_h[T - 1] = (
            model.drop(relevant_h[T - 1]),
            model.drop(irrelevant_h[T - 1]),
        )
    if not args.mean_hidden:
        scores = torch.matmul(W_out, relevant_h[T - 1])
        irrel_scores = torch.matmul(W_out, irrelevant_h[T - 1])
    else:
        mean_hidden = torch.mean(relevant_h, 0)  # [H]
        scores = torch.matmul(W_out, mean_hidden)
        irrel_scores = torch.matmul(W_out, torch.mean(irrelevant_h, 0))

    states = {"i": i_states, "g": g_states, "f": f_states, "c": c_states, "o": o_states}

    # if any(np.isnan(scores)):
    #     print(1)

    return scores, irrel_scores, states


def torch_mul(param, h):
    param = param.unsqueeze(0)  # [1, h1, h2]
    h = h.unsqueeze(-1)  # [B, h2, 1]
    mult = torch.matmul(param, h)  # [B, h1, 1]
    return mult.squeeze(-1)


def get_lstm_states(batch, model, gpu):
    if not args.task == "tacred":
        word_vecs = model.embed(batch.text)
        lstm_module = model.lstm
    else:
        token_vec = model.embed(batch.text)
        pos_vec = model.pos_embed(batch.pos)
        ner_vec = model.ner_embed(batch.ner)
        word_vecs = torch.cat([token_vec, pos_vec, ner_vec], -1)
        word_vecs = model.drop(word_vecs)
        lstm_module = model.lstm.rnn

    batch_size = word_vecs.size(1)
    hidden_dim = model.hidden_dim
    T = word_vecs.size(0)
    prev_c, prev_h = (
        torch.zeros(batch_size, hidden_dim).to(gpu),
        torch.zeros(batch_size, hidden_dim).to(gpu),
    )
    W_ii, W_if, W_ig, W_io = torch.split(
        lstm_module.weight_ih_l0, lstm_module.hidden_size
    )
    W_hi, W_hf, W_hg, W_ho = torch.split(
        lstm_module.weight_hh_l0, lstm_module.hidden_size
    )
    b_i, b_f, b_g, b_o = torch.split(
        lstm_module.bias_ih_l0 + lstm_module.bias_hh_l0, lstm_module.hidden_size
    )

    i_states, g_states, f_states, c_states = [], [], [], []
    act_i_states, act_g_states, act_f_states = [], [], []
    o_states = []
    temp_c_states = []
    tanhc_states = []

    for ts in range(T):
        i = torch_mul(W_hi, prev_h)
        g = torch_mul(W_hg, prev_h)
        f = torch_mul(W_hf, prev_h)

        i += torch_mul(W_ii, word_vecs[ts]) + b_i.view(1, -1)
        g += torch_mul(W_ig, word_vecs[ts]) + b_g.view(1, -1)
        f += torch_mul(W_if, word_vecs[ts]) + b_f.view(1, -1)

        i_states.append(i)
        g_states.append(g)
        f_states.append(f)

        act_i_states.append(sigmoid(i))
        act_g_states.append(tanh(g))
        act_f_states.append(sigmoid(f))
        temp_c_states.append(prev_c)

        c = sigmoid(f) * prev_c + sigmoid(i) * tanh(g)
        o = sigmoid(
            torch_mul(W_io, word_vecs[ts]) + torch_mul(W_ho, prev_h) + b_o.view(1, -1)
        )
        c_states.append(c)
        tanhc_states.append(tanh(c))
        o_states.append(o)

        h = o * tanh(c)

        prev_h = h
        prev_c = c

    states = {
        "i": i_states,
        "g": g_states,
        "f": f_states,
        "c": c_states,
        "o": o_states,
        "act_i": act_i_states,
        "act_g": act_g_states,
        "act_f": act_f_states,
        "temp_c": temp_c_states,
        "tanhc": tanhc_states,
    }

    return states


def decomp_activation_two_with_states(a, b, activation, states, state_key, t):
    rel = 0
    if states is None or not states["c"]:
        exemplar_size = 0
    else:
        exemplar_size = states["c"][0].size(0)
    for idx in range(exemplar_size):
        hs = states[state_key][t][idx]
        rel += activation(hs) - activation(hs - a)

    rel += activation(a + b) - activation(b)
    rel /= exemplar_size + 1

    irrel = activation(a + b) - rel
    return rel, irrel


def decomp_activation_three_with_states(
    rel_x, irrel_x, bias_x, activation, states, state_key, t, bias_in_rel
):
    rel = 0
    if states is None or not states["c"]:
        exemplar_size = 0
    else:
        exemplar_size = states["c"][0].size(0)

    for idx in range(exemplar_size):
        hs = states[state_key][t][idx]
        rel += activation(hs) - activation(hs - rel_x)

    rel += activation(rel_x + irrel_x + bias_x) - activation(irrel_x + bias_x)
    rel /= exemplar_size + 1

    bias = activation(bias_x)
    irrel = activation(rel_x + irrel_x + bias_x) - rel - bias

    return rel, irrel, bias


def mult_terms(
    rel_a,
    irrel_a,
    bias_a,
    rel_b,
    irrel_b,
    bias_b,
    states,
    state_key_a,
    state_key_b,
    t,
    bias_in_rel,
):
    rel = 0

    if states is None or not states["c"]:
        exemplar_size = 0
    else:
        exemplar_size = states["c"][0].size(0)

    for idx in range(exemplar_size):
        hs_a = states[state_key_a][t][idx]
        hs_b = states[state_key_b][t][idx]
        rel += rel_a * (hs_b - rel_b - bias_b) + rel_b * (hs_a - rel_a - bias_a)
    rel += rel_a * irrel_b + rel_b * irrel_a
    rel /= exemplar_size + 1
    rel += rel_a * rel_b + rel_a * bias_b + rel_b * bias_a

    irrel = (rel_a + irrel_a + bias_a) * (rel_b + irrel_b + bias_b) - rel
    return rel, irrel
