# -*- coding: utf-8 -*-

import logging
import multiprocessing as mp
import os
import sys
import warnings
import datetime
import logging
import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
from IPython import get_ipython

RUNNER_LOGGER_NAME = "TESS-ATLAS-RUNNER"
NOTEBOOK_LOGGER_NAME = "TESS-ATLAS"


class DeltaTimeFormatter(logging.Formatter):
    def format(self, record):
        duration = datetime.datetime.utcfromtimestamp(
            record.relativeCreated / 1000
        )
        record.delta = duration.strftime("%H:%M:%S")
        return super().format(record)


def setup_logger(logger_name, outdir=""):
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    # add custom formatter to root logger
    handler = logging.StreamHandler()
    formatter = DeltaTimeFormatter(
        "\033[92m[%(delta)s - %(name)s]\033[0m %(message)s"
    )

    # console logging
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    if outdir != "":  # setup file logging
        # create log file
        os.makedirs(outdir, exist_ok=True)
        fname = logger_name.replace("-", "_").lower()
        filename = os.path.join(outdir, f"{fname}.log")

        # setup log-file handler
        fh = logging.FileHandler(filename)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
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
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    set_plotting_style()
    set_theano_cache()


def set_plotting_style():
    plt.style.use("default")
    plt.rcParams["savefig.dpi"] = 100
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["font.size"] = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
    plt.rcParams["font.cursive"] = ["Liberation Sans"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["image.cmap"] = "inferno"


def set_theano_cache():
    # make sure that THEANO has cache dir for each thread (prevent locking issues)
    theano_cache = os.path.join(
        get_cache_dir(), "theano_cache", str(os.getpid())
    )
    os.makedirs(theano_cache, exist_ok=True)
    os.environ["THEANO_FLAGS"] = f"compiledir={theano_cache}"


def get_cache_dir(default="./"):
    # ozstar specific scratch space
    return os.environ.get("JOBFS", default=default)


def get_notebook_logger(outdir=""):
    # Logging setup
    for logger_name in [
        "theano.gof.compilelock",
        "filelock",
        "lazylinker_c.py",
        "theano.tensor.opt",
        "exoplanet",
        "matplotlib",
        "urllib3",
        "arviz",
        "astropy",
        "lightkurve",
        "corner",
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)

    notebook_logger = setup_logger(NOTEBOOK_LOGGER_NAME, outdir)
    return notebook_logger


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)
