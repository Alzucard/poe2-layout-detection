import json

import pytest

from poe2layout.files import read_json
from poe2layout.zones import Instance, ZoneData, is_town, normalize, zone_key


def test_normalize_and_zone_key():
    assert normalize("  The  Grelwood! ") == "the grelwood"
    assert normalize("Lost’s Rest") == "losts rest"
    assert zone_key("The Excavation") == "excavation"
    assert zone_key("Theatre") == "theatre"


def test_instance():
    inst = Instance("G1_2", 1628456714)
    assert inst.id == "G1_2_1628456714"
    assert Instance.from_json(inst.to_json()) == inst
    assert Instance.from_json(json.loads(json.dumps(inst.to_json()))) == inst


@pytest.mark.parametrize("zone, code, town", [
    ("Clearfell Encampment", None, True),
    ("Kingsmarch", None, True),
    ("Felled Hideout", None, True),
    ("Clearfell", None, False),
    ("Some Zone", "G1_town", True),
    ("Shrine", "HideoutFelled", True),
    ("Clearfell", "G1_2", False),
    (None, None, False),
])
def test_is_town(zone, code, town):
    assert is_town(zone, Instance(code, 1) if code else None) is town


@pytest.fixture
def zones_file(tmp_path):
    p = tmp_path / "zones.json"
    p.write_text(json.dumps({"Act 1": [
        {"name": "Clearfell", "connections": ["Clearfell Encampment", "The Grelwood"],
         "points_of_interest": [{"name": "Beira of the Rotten Pack"}]},
        {"name": "The Grelwood", "connections": []},
    ]}), encoding="utf-8")
    return p


def test_gates_are_symmetric(zones_file):
    z = ZoneData(zones_file)
    assert z.exits("Clearfell") == {"clearfell encampment", "grelwood"}
    assert z.exits("The Grelwood") == {"clearfell"}           # only listed on Clearfell's side
    assert z.exits("Clearfell Encampment") == {"clearfell"}
    assert z.points_of_interest("clearfell") == ["Beira of the Rotten Pack"]
    assert z.exits(None) == set() and z.exits("Unknown") == set()


def test_learn_exit_saves_both_directions(zones_file, tmp_path):
    learned = tmp_path / "data" / "learned_exits.json"
    z = ZoneData(zones_file, learned)
    assert z.learn_exit("Red Vale", "The Grelwood")
    assert "red vale" in z.exits("The Grelwood")
    assert z.exits("Red Vale") == {"grelwood"}
    assert read_json(learned) == {"grelwood": ["Red Vale"], "red vale": ["The Grelwood"]}
    assert not z.learn_exit("Red Vale", "The Grelwood")       # nothing new
    assert not z.learn_exit("Clearfell", "The Grelwood")      # already in zones.json
    # a new ZoneData reads the learned gates back in
    assert ZoneData(zones_file, learned).exits("Red Vale") == {"grelwood"}


@pytest.mark.parametrize("zone, came_from", [
    ("Clearfell", "Clearfell"),
    ("Felled Hideout", "Clearfell"),
    ("Clearfell", "Felled Hideout"),
    ("Clearfell", None),
])
def test_learn_exit_ignores(zones_file, zone, came_from):
    assert not ZoneData(zones_file).learn_exit(zone, came_from)


def test_missing_or_corrupt_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    z = ZoneData(tmp_path / "none.json", bad)
    assert z.gates("Clearfell") == []
