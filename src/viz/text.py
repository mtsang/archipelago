import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import viz.colors as colors
import matplotlib.patches as patches
from viz.rec import _set_axis_config


stixfont = {"fontname": "STIXGeneral"}


def get_fig(fig=None, figsize=None, fontsize=10):
    def auto_figsize(fontsize):
        return (12.8 * fontsize / 12, 1.71429 * fontsize / 12)

    if fig is None:
        if figsize is None:
            figsize = auto_figsize(fontsize)
        fig = plt.figure(figsize=figsize)
    return fig


def viz_text(
    explanation,
    tokens,
    fig=None,
    figsize=None,
    axis=None,
    shift=0,
    fontweight=500,
    fontsize=12,
    spacing=0.018,
    empty_spacing=0.007,
    cbar_pos=None,
    cbar_fontsize=20,
    show_colorbar=False,
    max_magn=None,
    size_ratio=None,
    clearfig=True,
    **kwargs,
):
    fig = get_fig(fig=fig, figsize=figsize, fontsize=fontsize)
    if clearfig:
        plt.clf()
    if axis is None:
        axis = fig.gca()

    if size_ratio is None:
        size_ratio = fontsize / 16

    plt.axis("off")
    axis_transform = axis.transData

    # normalize based on this bound
    if max_magn is None:
        max_magn = np.max(np.abs(list(explanation.values())))

    normalizer = mpl.colors.Normalize(vmin=-max_magn, vmax=max_magn)
    color_mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap=colors.pos_neg_colors())

    # sets attribution for interaction elements. notes any contiguous interaction tokens for display purposes
    interaction_effects = {}
    min_k = np.Inf
    for k in explanation:
        assert isinstance(k, tuple)
        if len(k) > 1:
            interaction_effects[k] = explanation[k]
        min_k = min(min(k), min_k)

    interaction_token_effect = {}
    interaction_next_token = set()
    for inter in interaction_effects:
        for i in inter:
            interaction_token_effect[i] = interaction_effects[inter]
            if i + 1 in inter:
                interaction_next_token.add(i)

    token_pos = {}

    x_pos = 0

    for t, token in enumerate(tokens):

        k = t + min_k

        token = token.replace("##", "-")
        y_pos = 0.5
        zorder = 0

        if k in interaction_token_effect:
            importance = interaction_token_effect[k]
        else:
            importance = explanation[(k,)]

        color = color_mapper.to_rgba(importance)
        text = plt.text(
            x=x_pos,
            y=y_pos + shift,
            s="{}".format(token),
            backgroundcolor=color,
            fontsize=fontsize,
            transform=axis_transform,
            fontweight=fontweight,
            zorder=zorder,
        )

        text.draw(fig.canvas.get_renderer())
        ex = text.get_window_extent()

        # the horizontal token position in the middle of the token "window"
        x = (ex.p0[0] + ex.p1[0]) / 2
        # the vertical token position at the top of the same window
        y = ex.p0[1]

        inv_axis = axis.transData.inverted()

        # the position of the token on the axis
        token_pos[k] = tuple(inv_axis.transform((x, y)))

        if k in interaction_next_token:
            added_spacing = empty_spacing
        else:
            added_spacing = spacing

        x_pos = inv_axis.transform(ex.p1)[0] + added_spacing

    draw_interaction_arrows(
        axis, interaction_effects, token_pos, fontsize, size_ratio, **kwargs
    )
    if show_colorbar and cbar_pos is not None:
        cbar_ax = fig.add_axes(cbar_pos)
        color_bar = plt.colorbar(
            color_mapper,
            cax=cbar_ax,
            orientation="vertical",
            ticks=[-max_magn, max_magn],
        )
        color_bar.ax.tick_params(size=0)
        color_bar.ax.set_yticklabels(
            ["neg", "pos"], fontsize=cbar_fontsize, **stixfont
        )  # vertically oriented colorbar
        #         color_bar.set_label('Attribution Value', fontsize=16)
        color_bar.outline.set_visible(False)

    return fig


def draw_interaction_arrows(
    axis,
    interaction_effects,
    token_pos,
    fontsize,
    size_ratio,
    arrow_shift=0.2,
    arrow_base_level=20,
    arrow_ext_const=6.5,
    arrow_head_width=3,
    arrow_head_length=6,
    arrow_linewidth=1,
):
    if len(interaction_effects) == 1:
        if len(list(interaction_effects.keys())[0]) == len(token_pos):
            return

    arrow_head_width *= size_ratio
    arrow_head_length *= size_ratio
    arrow_shift = 0.18 + (arrow_shift - 0.2) * size_ratio
    arrow_linewidth *= size_ratio
    arrow_base_level = 4 + (arrow_base_level - 4) * size_ratio
    arrow_ext_const *= size_ratio
    style = (
        "<|-|>,head_width="
        + str(arrow_head_width)
        + ",head_length="
        + str(arrow_head_length)
    )  # +",linewidth=0.5" #tail_width=0.5,
    kw = dict(arrowstyle=style, color="k", lw=arrow_linewidth)

    used_ext = set()
    first_overlap_found = False
    found_loop = -1

    for i, inter in enumerate(interaction_effects):
        ext = 0
        for inter2 in interaction_effects:
            if inter != inter2:
                if inter[0] < inter2[0] and inter2[-1] < inter[-1]:

                    ext = len(used_ext) + 1
                    used_ext.add(ext)

                    if found_loop == i:
                        first_overlap_found = False
                    break
                elif any(
                    inter[0] < v < inter[-1] for v in inter2
                ):  # TODO: what about multiple overlap cases?
                    if first_overlap_found:
                        ext = len(used_ext) + 1
                        used_ext.add(ext)
                        break
                    else:
                        first_overlap_found = True
                        found_loop = i

        for j in range(len(inter) - 1):
            a, b = inter[j : j + 2]
            pa = token_pos[a]
            pb = token_pos[b]
            h = arrow_base_level + arrow_ext_const * ext

            if b == a + 1:
                rad = 10
            else:
                rad = 15

            rad *= size_ratio

            cstyle = patches.ConnectionStyle.Arc(
                armA=h, armB=h, angleA=90, angleB=90, rad=rad
            )
            arrow = patches.FancyArrowPatch(
                (pa[0], pa[1] + arrow_shift),
                (pb[0], pb[1] + arrow_shift),
                connectionstyle=cstyle,
                **kw,
            )
            axis.add_patch(arrow)


def interactive_viz_text(
    exps,
    tokens,
    process_stop_words,
    init_k=3,
    fontsize=10,
    fig=None,
    figsize=None,
    max_magn=None,
    **kwargs,
):
    from ipywidgets import widgets, interact

    fig = get_fig(fontsize=fontsize)

    def update(k=init_k):
        explanation = exps[k]
        explanation_p, tokens_p = process_stop_words(explanation, tokens)
        viz_text(
            explanation_p,
            tokens_p,
            fig=fig,
            max_magn=max_magn,
            fontsize=fontsize,
            arrow_base_level=25,
            **kwargs,
        )
        fig.canvas.draw_idle()

    interact(
        update, k=widgets.IntSlider(value=init_k, min=0, max=len(exps) - 1, step=1)
    )
