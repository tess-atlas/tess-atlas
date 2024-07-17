import sys

import pytest

from tess_atlas.cli import (
    download_toi_cli,
    run_toi_cli,
    tess_atlas_summary_cli,
)


def test_run_toi_cli(monkeypatch):
    with pytest.raises(SystemExit):
        monkeypatch.setattr(sys, "argv", ["cmd", "--help"])
        run_toi_cli.main()


def test_tess_atlas_summary_cli(monkeypatch):
    with pytest.raises(SystemExit):
        monkeypatch.setattr(sys, "argv", ["cmd", "--help"])
        tess_atlas_summary_cli.main()


def test_download_toi_cli(monkeypatch):
    with pytest.raises(SystemExit):
        monkeypatch.setattr(sys, "argv", ["cmd", "--help"])
        download_toi_cli.main()
