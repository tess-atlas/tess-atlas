# -*- coding: utf-8 -*-

import logging
import multiprocessing as mp
import os
import sys
import warnings

import matplotlib.pyplot as plt
from IPython import get_ipython

RUNNER_LOGGER_NAME = "TESS-ATLAS-RUNNER"
NOTEBOOK_LOGGER_NAME = "TESS-ATLAS"


def setup_logger(logger_name, outdir=""):
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(sh)
    if outdir != "":
        os.makedirs(outdir, exist_ok=True)
        filename = os.path.join(outdir, f"{logger_name}_runner.log")
        fh = logging.FileHandler(filename)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def notebook_initalisations():
    get_ipython().magic('config InlineBackend.figure_format = "retina"')

    try:
        mp.set_start_method("fork")
    except RuntimeError:  # "Multiprocessing context already set"
        pass

    # Don't use the schmantzy progress bar
    os.environ["EXOPLANET_NO_AUTO_PBAR"] = "true"

    # Warning
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    plt.style.use("default")
    plt.rcParams["savefig.dpi"] = 100
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["font.size"] = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
    plt.rcParams["font.cursive"] = ["Liberation Sans"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["image.cmap"] = "inferno"

    # make sure that THEANO has cache dir for each thread (prevent locking issues)
    os.environ["THEANO_FLAGS"] = f"compiledir=./theano_cache/{os.getpid()}"


def get_logger():
    # Logging setup
    for logger_name in [
        "theano.gof.compilelock",
        "exoplanet",
        "matplotlib",
        "urllib3",
        "arviz",
        "astropy",
        "lightkurve",
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)

    notebook_logger = setup_logger(NOTEBOOK_LOGGER_NAME)
    notebook_logger.setLevel(logging.INFO)
    return notebook_logger
