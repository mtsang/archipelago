import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import viz.colors as colors
from textwrap import wrap


def viz_bar_chart(
    data,
    top_k=5,
    figsize=(10, 4),
    save_file=None,
    max_label_size=100,
    remove_space=False,
    y_label="Feature Set",
    sort_again=True,
    bounds=None,
    **kwargs
):

    feature_labels, attributions = zip(*data)
    if top_k > len(attributions):
        top_k = len(attributions)

    args = np.argsort(-1 * np.abs(attributions))
    args = args[:top_k]

    args2 = np.argsort(np.array(attributions)[args])
    feature_labels = np.array(feature_labels)[args]
    attributions = np.array(attributions)[args]

    if sort_again:
        feature_labels = feature_labels[args2]
        attributions = attributions[args2]

    fig, axis = plt.subplots(figsize=figsize)

    if bounds is None:
        bounds = np.max(np.abs(attributions))
    normalizer = mpl.colors.Normalize(vmin=-bounds, vmax=bounds)

    if "cmap" in kwargs:
        cmap = kwargs["cmap"]
    else:
        cmap = colors.pos_neg_colors()

    axis.barh(
        np.arange(top_k),
        attributions,
        color=[cmap(normalizer(c)) for c in attributions],
        align="center",
        zorder=10,
        **kwargs
    )

    if not sort_again:
        axis.invert_yaxis()

    axis.set_xlabel("Attribution", fontsize=18)
    axis.set_ylabel(y_label, fontsize=18)
    axis.set_yticks(np.arange(top_k))
    axis.tick_params(axis="y", which="both", left=False, labelsize=14)
    axis.tick_params(axis="x", which="both", left=False, labelsize=14)

    if remove_space:
        token = " "
    else:
        token = ""

    axis.set_yticklabels(
        ["\n".join(wrap(y, max_label_size)).replace(token, "") for y in feature_labels]
    )

    axis.grid(axis="x", zorder=0, linewidth=0.2)
    axis.grid(axis="y", zorder=0, linestyle="--", linewidth=1.0)
    _set_axis_config(axis, linewidths=(0.0, 0.0, 0.0, 1.0))
    if save_file is not None:
        plt.savefig(save_file, bbox_inches="tight")


def _set_axis_config(
    axis, linewidths=(0.0, 0.0, 0.0, 0.0), clear_y_ticks=False, clear_x_ticks=False
):
    """
    Source: Integrated Hessians Code Repo
    """
    axis.spines["right"].set_linewidth(linewidths[0])
    axis.spines["top"].set_linewidth(linewidths[1])
    axis.spines["left"].set_linewidth(linewidths[2])
    axis.spines["bottom"].set_linewidth(linewidths[3])
    if clear_x_ticks:
        axis.set_xticks([])
    if clear_y_ticks:
        axis.set_yticks([])
