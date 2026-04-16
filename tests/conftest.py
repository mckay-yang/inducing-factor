import tempfile

import pytest

from inducing_factor.utils import build_synthetic_panel, save_as_npz_chunks


@pytest.fixture
def synthetic_cache_and_output():
    with tempfile.TemporaryDirectory() as cache_dir, tempfile.TemporaryDirectory() as out_dir:
        save_as_npz_chunks(build_synthetic_panel(), cache_dir)
        yield cache_dir, out_dir
