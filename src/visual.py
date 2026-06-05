import curses
from time import sleep
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sparse
from IPython.display import HTML, display
from matplotlib import animation
from numpy.typing import NDArray
from scipy.spatial.distance import squareform


def plot_adj(
    adjacencies: NDArray,
    height=3,
    highlight=None,
    titles: Optional[list] = None,
    axes=None,
    **kwargs,
):
    """Plot a sequence of adjacency matrices.
    Args:
        adjacencies (_type_): Sequence of adjacency matrices.
        highlight (_type_, optional): List of indices to highlight. Defaults to None.

    Returns:
        _type_: _description_
    """
    n_steps = len(adjacencies)
    width = n_steps * 2
    if titles is not None:
        height *= 1.05

    if axes is None:
        fig, axes = plt.subplots(1, n_steps, figsize=(width, height), dpi=200)
        if n_steps == 1:
            axes = [axes]
    else:
        fig = None

    for i, (ax, adj) in enumerate(zip(axes, adjacencies)):
        ax.imshow(adj, vmin=0, vmax=1, **kwargs)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        if highlight is not None and i in highlight:
            for spine in ax.spines.values():
                spine.set_edgecolor("tab:orange")
                spine.set_linewidth(3)

        if titles is not None:
            ax.set_title(titles[i])

    return fig, axes


def plot_graph(
    adjacencies: NDArray,
    height=3,
    highlight=None,
    titles: Optional[list] = None,
    axes=None,
    eps=1e-8,
):
    """Plot graphs from adjacency matrices.

    Args:
        adjacencies (_type_): _description_
        height (int, optional): _description_. Defaults to 3.
        highlight (_type_, optional): _description_. Defaults to None.
        titles (Optional[list], optional): _description_. Defaults to None.
        axes (_type_, optional): _description_. Defaults to None.
    """

    n_steps, n_nodes, _ = adjacencies.shape
    width = n_steps * height
    if titles is not None:
        height *= 1.05

    if axes is None:
        fig, axes = plt.subplots(1, n_steps, figsize=(width, height), dpi=200)
        if n_steps == 1:
            axes = [axes]
    else:
        fig = None

    for i, (ax, adj) in enumerate(zip(axes, adjacencies / adjacencies.max())):
        g = nx.from_numpy_array(adj > eps)
        edge_list = [adj[e].item() for e in g.edges()]
        nx.draw_circular(
            g,
            node_size=50 * height / n_nodes,
            width=3 * height / n_nodes,
            edge_color=edge_list,
            edge_cmap=plt.cm.Greys,
            edge_vmin=0,
            edge_vmax=1,
            ax=ax,
        )

        if highlight is not None and i in highlight:
            for spine in ax.spines.values():
                spine.set_edgecolor("tab:orange")
                spine.set_linewidth(3)

        if titles is not None:
            ax.set_title(titles[i])

    return fig, axes


def print_sequence(snapshots: NDArray, title: str):
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()

    for i, snap in enumerate(snapshots):
        stdscr.addstr(0, 0, f"{title} - Sample {i}")
        stdscr.addstr(1, 0, "\n".join(" ".join("*" if el else " " for el in row) for row in snap))
        stdscr.refresh()
        sleep(0.2)

    curses.echo()
    curses.nocbreak()
    curses.endwin()


def interpolation_video(
    t: NDArray,
    *interpolations: sparse.COO | NDArray,
    ax_width=2,
    video_len=5000,
    max_ncols=8,
    **named_interpolations: sparse.COO | NDArray,
) -> animation.FuncAnimation:
    """Animate adjacency matrix evolution of interpolation.

    Args:
        t (NDArray): Time points of the interpolation
        interpolation (NDArray): Tensor of adjcency matrices of shape (n_samples, n_nodes, n_nodes)
    """
    named_interpolations.update({f"Seq {i}": interp for i, interp in enumerate(interpolations)})

    if (n_inter := len(named_interpolations)) < 2:
        fig, axes = plt.subplots(1, 2, figsize=(2 * ax_width, ax_width), dpi=200)
        axes = axes[:, None]
    else:
        n_cols = min(max_ncols, n_inter)
        n_rows = 2 * int(np.floor(n_inter / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * ax_width, n_rows * ax_width), dpi=200
        )

    n_nodes = named_interpolations[list(named_interpolations)[0]].shape[1]

    def animate(iteration):
        for ax in axes.flatten():
            ax.clear()
        for (ax0, ax1), (name, interpolation) in zip(
            axes.T.reshape(-1, 2), named_interpolations.items()
        ):
            weight = interpolation[iteration]

            if isinstance(weight, sparse.SparseArray):
                weight = weight.todense()

            ax0.imshow(weight)
            ax0.set(title=name, xticklabels=[], yticklabels=[])
            g = nx.from_numpy_array(weight)
            edge_list = squareform(weight, checks=False)
            edge_list = edge_list[edge_list > 0]
            nx.draw_circular(g, node_size=300 / n_nodes, width=edge_list * 15 / n_nodes, ax=ax1)

        fig.suptitle(f"Step {iteration}/{len(t)-1}")
        fig.tight_layout()

    anim = animation.FuncAnimation(
        fig, animate, frames=range(len(t)), interval=video_len / (len(t) - 1)
    )
    plt.close()
    return anim


def display_animation(video: animation.FuncAnimation, embed_limit: float = None):
    display(HTML(video.to_html5_video(embed_limit=embed_limit)))
