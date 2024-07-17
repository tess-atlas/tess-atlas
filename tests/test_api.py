import logging

import pytest

from tess_atlas import __website__
from tess_atlas.api import download_analysed_toi
from tess_atlas.logger import LOGGER_NAME


@pytest.fixture(autouse=True)
def monkeypatch_command(monkeypatch):
    monkeypatch.setattr(
        download_analysed_toi,
        "COMMAND",
        f"echo {download_analysed_toi.COMMAND}",
    )


def test_download_toi(caplog):
    caplog.set_level(logging.DEBUG)
    logging.getLogger(LOGGER_NAME).setLevel(logging.DEBUG)
    download_analysed_toi.download_toi(123)

    assert __website__ in caplog.text
    assert "toi_123" in caplog.text
