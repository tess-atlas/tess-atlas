import logging
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from ..utils import NOTEBOOK_LOGGER_NAME
from .labels import LATEX, PARAMS_CATEGORIES, PRIOR_PLOT
from .plotting_utils import (
    exception_catcher,
    format_label_string_with_offset,
    format_prior_samples_and_initial_params,
)

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


@exception_catcher
def plot_priors(
    tic_entry: "TICEntry", prior_samples: Dict, init_params: Dict
) -> None:
    prior_samples, init_params = format_prior_samples_and_initial_params(
        prior_samples, init_params
    )

    samples_table = {}
    samples_table["Noise Params"] = get_samples_from_param_regexs(
        prior_samples, PARAMS_CATEGORIES["NOISE PARAMS"]
    )
    samples_table["Stellar Params"] = get_samples_from_param_regexs(
        prior_samples, PARAMS_CATEGORIES["STELLAR PARAMS"]
    )
    samples_table[f"Period Params"] = get_samples_from_param_regexs(
        prior_samples, PARAMS_CATEGORIES["PERIOD PARAMS"]
    )
    samples_table[f"Planet Params"] = get_samples_from_param_regexs(
        prior_samples, PARAMS_CATEGORIES["PLANET PARAMS"]
    )

    try:
        fig = plot_histograms(
            samples_table, trues=init_params, latex_label=LATEX
        )

        fname = os.path.join(tic_entry.outdir, f"{PRIOR_PLOT}")
        logger.debug(f"Saving {fname}")
        fig.savefig(fname)
    except Exception as e:
        logger.error(f"Cant plot priors: {e}")


def get_samples_from_param_regexs(samples, param_regex):
    samples_keys = samples.columns.values
    data = {}
    for s in samples_keys:
        for regex in param_regex:
            if regex == s or f"{regex}_" in s:
                data[s] = samples[s]
    return data


def plot_histograms(
    samples_table: Dict[str, Dict[str, np.array]],
    fname: Optional[str] = "",
    trues: Optional[Dict] = {},
    latex_label: Optional[Dict] = {},
) -> None:
    nrows = len(samples_table.keys())
    ncols = __get_longest_row_length(samples_table)
    fig, axes = __create_fig(nrows, ncols)

    for row_i, (set_label, sample_set) in enumerate(samples_table.items()):
        axes[row_i, 0].set_title(set_label + ":", loc="left")
        for col_i, (sample_name, samples) in enumerate(sample_set.items()):
            __plot_hist1d(axes[row_i, col_i], samples)

            if trues:
                axes[row_i, col_i].axvline(trues[sample_name], color="C1")
            __add_ax(axes[row_i, col_i])
            axes[row_i, col_i].set_xlabel(
                latex_label.get(sample_name, sample_name)
            )
            format_label_string_with_offset(axes[row_i, col_i], "x")
    plt.tight_layout()
    if fname:
        fig.savefig(fname)
    else:
        return fig


def __plot_hist1d(ax, x):
    range = np.quantile(x, [0.01, 0.99])
    ax.hist(x, density=True, bins=20, range=range, histtype="step", color="C0")
    ax.set_xlim(min(range), max(range))


def __create_fig(nrows, ncols):
    xdim, ydim = ncols * 2.5, nrows * 2.5
    fig, axes = plt.subplots(nrows, ncols, figsize=(xdim, ydim))
    for i in range(nrows):
        for j in range(ncols):
            __remove_ax(axes[i, j])
    return fig, axes


def __remove_ax(ax):
    ax.set_yticks([])
    ax.set_xticks([])
    ax.tick_params(direction="in")
    ax.set_frame_on(False)


def __add_ax(ax):
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(direction="in")
    ax.set_frame_on(True)


def __get_longest_row_length(
    samples_table: Dict[str, Dict[str, np.array]]
) -> int:
    return max(
        [
            len(samples_dicts.keys())
            for label, samples_dicts in samples_table.items()
        ]
    )
