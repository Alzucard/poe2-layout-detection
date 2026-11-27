import tomllib
from pathlib import Path

import poe2layout


def test_version_matches_pyproject():
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert poe2layout.__version__ == pyproject["project"]["version"]
