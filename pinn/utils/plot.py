import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def figsize(scale, nplots = 1):
    fig_width_pt = 390.0
    inches_per_pt = 1.0/72.27
    golden_mean = (np.sqrt(5.0)-1.0)/2.0
    fig_width = fig_width_pt*inches_per_pt*scale
    fig_height = nplots*fig_width*golden_mean
    fig_size = [fig_width,fig_height]
    return fig_size


pgf_with_latex = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(1.0),
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        ]
}


def newfig(width, nplots = 1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(fig: Figure, filename: str):
    fig.savefig(filename, dpi=fig.dpi)
