import numpy as np
import pytest

from poe2layout.files import imread, imwrite, read_json, safe_name, write_json_atomic


@pytest.mark.parametrize("name, expected", [
    ("The Grelwood", "The Grelwood"),
    ('a<b>c:"d"/e\\f|g?h*', "a_b_c__d__e_f_g_h_"),
    ("trailing dot.", "trailing dot"),
    ("CON", "_CON"),
    ("nul.txt", "_nul.txt"),
    ("COM9", "_COM9"),
    ("lpt1.log", "_lpt1.log"),
    ("COM10", "COM10"),
    ("", "_"),
])
def test_safe_name(name, expected):
    assert safe_name(name) == expected


def test_json_roundtrip_and_no_tmp_left(tmp_path):
    p = tmp_path / "x.json"
    write_json_atomic(p, {"b": 1, "a": [1, 2]}, sort_keys=True)
    assert read_json(p) == {"a": [1, 2], "b": 1}
    write_json_atomic(p, {"c": 3})
    assert read_json(p) == {"c": 3}
    assert [f.name for f in tmp_path.iterdir()] == ["x.json"]


def test_read_json_default_and_corrupt(tmp_path):
    assert read_json(tmp_path / "none.json", default={}) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        read_json(bad)


def test_image_roundtrip_non_ascii_path(tmp_path):
    folder = tmp_path / "Zoné ä"
    folder.mkdir()
    img = np.random.default_rng(0).integers(0, 255, (20, 30, 3), dtype=np.uint8)
    assert imwrite(folder / "map.png", img)
    assert np.array_equal(imread(folder / "map.png"), img)
    assert [f.name for f in folder.iterdir()] == ["map.png"]          # no temporary file left behind


def test_imread_missing_is_none(tmp_path):
    assert imread(tmp_path / "missing.png") is None


def test_failed_replace_leaves_no_tmp(tmp_path, monkeypatch):
    import poe2layout.files as files

    def locked(src, dst):
        raise PermissionError("in use")

    monkeypatch.setattr(files.os, "replace", locked)
    with pytest.raises(PermissionError):
        imwrite(tmp_path / "map.png", np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(PermissionError):
        write_json_atomic(tmp_path / "x.json", {})
    assert list(tmp_path.iterdir()) == []
