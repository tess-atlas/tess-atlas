import logging
import os
import traceback

import interruptingcow
import nbformat
from nbconvert import HTMLExporter
from ploomber_engine import execute_notebook

from ..logger import LOGGER_NAME

__all__ = ["execute_ipynb"]

DAY_IN_SEC = 60 * 60 * 24


def execute_ipynb(
    notebook_filename: str,
    timeout=DAY_IN_SEC,
    save_profiling_data=True,
    **kwargs,
):
    """Executes a notebook and saves its executed version.

    It also caches some profiling data in the notebook metadata in the execution dir.
    And saves an HTML version of the notebook (if requested).

    :param notebook_filename: path of notebook to process
    :return: bool if notebook-preprocessing successful/unsuccessful
    """
    success = False
    runner_logger = logging.getLogger(LOGGER_NAME)
    runner_logger.info(f"Executing {notebook_filename}")
    if save_profiling_data is not False:
        profile_memory = True
        profile_runtime = True
    else:
        profile_memory = False
        profile_runtime = False
    try:
        with interruptingcow.timeout(timeout, exception=TimeoutError):
            run_path = os.path.dirname(notebook_filename)
            execute_notebook(
                input_path=notebook_filename,
                output_path=notebook_filename,
                cwd=run_path,
                save_profiling_data=save_profiling_data,
                profile_memory=profile_memory,
                profile_runtime=profile_runtime,
                log_output=False,
                debug_later=False,
            )
            success = True
    except Exception as e:
        err_str = traceback.format_exc()
        runner_logger.error(
            f"Preprocessing {notebook_filename} failed:\n{err_str}"
        )

    return success


def __read_ipynb_to_nbnode(
    notebook_filename: str,
) -> nbformat.notebooknode.NotebookNode:
    with open(notebook_filename) as f:
        return nbformat.read(f, as_version=4)
