from typing import List

import pandas as pd

from tess_atlas.data.exofop import EXOFOP_DATA
from tess_atlas.data.exofop.exofop_database import ExofopDatabase


def test_get_toi_list():
    toi_list = EXOFOP_DATA.get_toi_list()
    assert isinstance(toi_list, list)
    assert len(toi_list) > 0


def test_get_toi_numbers_for_different_categories():
    toi_numbers = EXOFOP_DATA.get_categorised_toi_lists()
    assert isinstance(toi_numbers.single_transit, List)
    assert len(toi_numbers.single_transit) > 0
    assert EXOFOP_DATA.get_counts() is not None
    print(EXOFOP_DATA)
    assert isinstance(EXOFOP_DATA.get_tic_data([101]), pd.DataFrame)


def test_download_if_not_present(tmpdir):
    ExofopDatabase(cache_fname=tmpdir / "test.csv", clean=True)
    ExofopDatabase(cache_fname=tmpdir / "test.csv", clean=False)
