from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import frame_at, make_world
from poe2layout.files import read_json
from poe2layout.library import META_VERSION, RunLibrary
from poe2layout.render import map_picture, render_run_images
from poe2layout.stitcher import Stitcher
from poe2layout.zones import Instance


def stitched(world, n=30, start=(500, 400)):
    s = Stitcher()
    for i in range(n):
        s.add(frame_at(world, start[0] + 25 * i, start[1] + 12 * i))
    return s


def test_every_floor_is_saved(world, tmp_path):
    s = stitched(world, 40)
    s.add(np.zeros((1080, 1920, 3), np.uint8))               # stairs: the screen fades to black
    for i in range(32):
        s.add(frame_at(make_world(seed=5), 500 + 25 * i, 400 + 12 * i))
    s.finalize()
    lib = RunLibrary(tmp_path / "runs")
    res = lib.save("Temple of Kopec", s, Instance("G3_12", 1), "Ziggurat Encampment")
    meta = read_json(res.folder / "meta.json")
    assert meta["frames"] == 40                               # map.png is the first floor, where you came in
    assert [(f["map"], f["frames"]) for f in meta["other_floors"]] == [("map_floor2.png", 32)]
    for f in ("map.png", "map_floor2.png", "path_floor2.png", "labels_floor2.png", "floors.png"):
        assert (res.folder / f).exists(), f
    floors = cv2.imread(str(res.folder / "floors.png"))
    first = cv2.imread(str(res.folder / "map.png"))
    assert floors.shape[0] > first.shape[0]                  # the floors stacked on top of each other


def test_save_and_reload(world, tmp_path):
    lib = RunLibrary(tmp_path / "runs", diagnostics_dir=tmp_path / "logs")
    inst = Instance("G1_2", 1628456714)
    res = lib.save("Clearfell", stitched(world), inst, "Clearfell Encampment", {"clearfell encampment"})
    assert res is not None
    assert res.folder == tmp_path / "runs" / "Clearfell" / "G1_2_1628456714"
    for f in ("map.png", "meta.json", "path.png", "labels.png"):
        assert (res.folder / f).exists(), f
    meta = read_json(res.folder / "meta.json")
    assert meta["version"] == META_VERSION
    assert (meta["zone"], meta["instance"], meta["zone_code"], meta["seed"]) == \
        ("Clearfell", "G1_2_1628456714", "G1_2", 1628456714)
    assert meta["came_from"] == "Clearfell Encampment"
    assert meta["screen"] == [1920, 1080]
    assert meta["entrance"] == [float(v) for v in meta["first_point"]]     # no labels read, so it's the first point
    assert lib.count("Clearfell") == 1
    runs = lib.runs("Clearfell")
    assert [r.id for r in runs] == ["G1_2_1628456714"]
    out = lib.layouts("Clearfell")
    assert out == {"G1_2_1628456714": {"layout": "A", "orientation": "id", "mirrored": False}}
    assert read_json(lib.layouts_path("Clearfell")) == out


def test_smaller_map_of_same_instance_is_not_saved(world, tmp_path):
    lib = RunLibrary(tmp_path / "runs")
    inst = Instance("G1_2", 1)
    assert lib.save("Clearfell", stitched(world, 30), inst, None) is not None
    assert lib.save("Clearfell", stitched(world, 5), inst, None) is None
    assert lib.count("Clearfell") == 1


def test_too_small_map_is_not_saved(tmp_path):
    lib = RunLibrary(tmp_path / "runs", diagnostics_dir=tmp_path / "logs")
    assert lib.save("Clearfell", Stitcher(), Instance("G1_2", 1), None) is None
    assert lib.count("Clearfell") == 0


def test_too_small_map_leaves_a_picture_for_you(world, tmp_path):
    lib = RunLibrary(tmp_path / "runs", diagnostics_dir=tmp_path / "logs")
    assert lib.save("Clearfell", stitched(world, 1), Instance("G1_2", 1), None) is None    # one frame, nothing stable
    assert [f.name.startswith("unsaved_Clearfell_") for f in (tmp_path / "logs").iterdir()] == [True]


def test_bigger_map_of_same_instance_is_sorted_again(world, tmp_path):
    lib = RunLibrary(tmp_path / "runs")
    inst = Instance("G1_2", 1)
    lib.save("Clearfell", stitched(world, 5), inst, None)
    assert inst.id in lib.layouts("Clearfell")
    assert lib.save("Clearfell", stitched(world, 30), inst, None) is not None
    assert inst.id not in read_json(lib.layouts_path("Clearfell"))      # the next layouts() compares it again
    assert inst.id in lib.layouts("Clearfell")


def test_most_recently_used_zones_stay_loaded(tmp_path):
    lib = RunLibrary(tmp_path / "runs", keep_zones=2)
    a = lib.runs("A")
    lib.runs("B")
    assert lib.runs("A") is a                                # used again, so now B is the oldest
    lib.runs("C")
    assert lib.runs("A") is a


def test_map_that_cannot_be_written_writes_no_meta(world, tmp_path, monkeypatch):
    import pytest

    import poe2layout.library as library
    monkeypatch.setattr(library, "imwrite", lambda path, img: False)
    lib = RunLibrary(tmp_path / "runs")
    with pytest.raises(OSError):
        lib.save("Clearfell", stitched(world), Instance("G1_2", 1), None)
    assert not (lib.run_dir("Clearfell", "G1_2_1") / "meta.json").exists()


def test_path_picture_leaves_a_gap_at_a_teleport(tmp_path):
    m = np.zeros((400, 800), np.uint8)
    walk = [(50 + 5 * i, 100) for i in range(20)] + [(600 + 5 * i, 300) for i in range(20)]    # a waypoint jump
    render_run_images(tmp_path, m, {"labels": []}, path=walk)
    img = cv2.imread(str(tmp_path / "path.png"))
    middle = img[195:206, 367:378]                   # halfway along the jump: no line drawn through the walls
    assert (middle == map_picture(m)[195:206, 367:378]).all()


def path_drawn(img, m, x0, y0, x1, y1):
    return (img[y0:y1, x0:x1] != map_picture(m)[y0:y1, x0:x1]).any()


def test_path_picture_from_a_thinned_route_stays_whole(tmp_path):
    # meta.json keeps every 10th point of a long visit, so 200 px between them is still walking
    m = np.zeros((400, 1200), np.uint8)
    walk = [(50 + 200 * i, 200) for i in range(5)]
    render_run_images(tmp_path, m, {"labels": [], "path": walk, "path_t": [10 * i for i in range(5)]})
    img = cv2.imread(str(tmp_path / "path.png"))
    assert path_drawn(img, m, 340, 195, 360, 206)


def test_path_picture_leaves_a_gap_where_frames_are_missing(tmp_path):
    # 10 s without a placed frame (lost, or a loading screen), so we don't know where you went in between
    m = np.zeros((400, 800), np.uint8)
    walk = [(100, 200), (110, 200), (220, 200), (230, 200)]
    render_run_images(tmp_path, m, {"labels": []}, path=walk, times=[1, 2, 22, 23])
    img = cv2.imread(str(tmp_path / "path.png"))
    assert not path_drawn(img, m, 160, 195, 171, 206)
    assert path_drawn(img, m, 101, 195, 109, 206)


def test_run_folder_named_by_the_first_version_is_found(world, tmp_path):
    lib = RunLibrary(tmp_path / "runs")
    inst = Instance("G1_2", 1)
    first = lib.save("Clearfell", stitched(world, 30), inst, None).folder
    old = first.with_name("A_1")                                    # named after its layout, like the first version did
    first.rename(old)
    assert lib.save("Clearfell", stitched(world, 5), inst, None) is None    # still the bigger map, so it's kept
    assert lib.count("Clearfell") == 1 and lib.run_dir("Clearfell", inst.id).exists() and not old.exists()


def test_save_cut_short_leaves_the_old_map_and_meta_together(world, tmp_path, monkeypatch):
    import poe2layout.library as library
    lib = RunLibrary(tmp_path / "runs")
    inst = Instance("G1_2", 1)
    folder = lib.save("Clearfell", stitched(world, 30), inst, None).folder
    before = {n: (folder / n).read_bytes() for n in ("map.png", "meta.json")}

    def killed(*args):
        raise KeyboardInterrupt                                    # the process ends while saving

    monkeypatch.setattr(library, "write_json_atomic", killed)
    with pytest.raises(KeyboardInterrupt):
        lib.save("Clearfell", stitched(world, 60), inst, None)
    assert {n: (folder / n).read_bytes() for n in before} == before


def test_save_cut_short_between_the_renames_is_finished_on_load(world, tmp_path):
    lib = RunLibrary(tmp_path / "runs")
    inst = Instance("G1_2", 1)
    folder = lib.save("Clearfell", stitched(world, 30), inst, None).folder
    folder.rename(folder.with_name(folder.name + ".new"))       # the old one moved away, the new one not in place yet
    assert [r.id for r in RunLibrary(tmp_path / "runs").runs("Clearfell")] == [inst.id]
    assert sorted(p.name for p in folder.parent.iterdir() if p.is_dir()) == [folder.name]


def test_unfinished_new_run_is_dropped(world, tmp_path):
    lib = RunLibrary(tmp_path / "runs")
    inst = Instance("G1_2", 1)
    folder = lib.save("Clearfell", stitched(world, 30), inst, None).folder
    half = folder.with_name(folder.name + ".new")
    half.mkdir()
    (half / "map.png").write_bytes(b"half")
    assert len(lib.runs("Clearfell")) == 1 and not half.exists()


def test_old_folder_that_cannot_be_removed_keeps_the_video_for_later(world, tmp_path, monkeypatch):
    # a file in <id>.old is still open (a viewer); the next save mustn't fail like a broken video (FileExistsError)
    import poe2layout.library as library
    lib = RunLibrary(tmp_path / "runs")
    inst = Instance("G1_2", 1)
    folder = lib.save("Clearfell", stitched(world, 30), inst, None).folder
    stuck = folder.with_name(folder.name + ".old")
    stuck.mkdir()
    (stuck / "path.png").write_bytes(b"open in a viewer")
    real = library.shutil.rmtree
    monkeypatch.setattr(library.shutil, "rmtree",
                        lambda p, *a, **k: None if Path(p) == stuck else real(p, *a, **k))
    with pytest.raises(PermissionError):
        lib.save("Clearfell", stitched(world, 60), inst, None)
    assert read_json(folder / "meta.json")["frames"] == 30


def test_count_changes_no_folders(world, tmp_path):
    lib = RunLibrary(tmp_path / "runs")
    folder = lib.save("Clearfell", stitched(world, 30), Instance("G1_2", 1), None).folder
    half = folder.with_name("G1_2_2.new")
    half.mkdir()
    (half / "meta.json").write_text("{}", encoding="utf-8")
    assert lib.count("Clearfell") == 1 and half.exists()
