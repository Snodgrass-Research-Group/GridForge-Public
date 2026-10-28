"""migrate() must not touch an existing data_store when its input is wrong."""

import pytest

from gridforge.io.matlab_import import migrate


def test_wrong_matlab_root_leaves_store_untouched(tmp_path):
    store = tmp_path / "data_store"
    store.mkdir()
    (store / "keep.parquet").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="missing"):
        migrate(tmp_path / "not-the-matlab-tree", store, overwrite=True)
    assert (store / "keep.parquet").read_bytes() == b"x"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data_store"]
