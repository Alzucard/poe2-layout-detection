import json

import pytest

from poe2layout.config import PROJECT_ROOT, Config, ConfigError, config_from_dict, load_config


def test_defaults_when_no_file(tmp_path):
    cfg, problems = load_config(tmp_path / "missing.json")
    assert cfg == Config()
    assert problems == []


def test_paths_follow_data_dir(tmp_path):
    cfg, _ = config_from_dict({"data_dir": str(tmp_path)})
    assert cfg.runs_dir == tmp_path / "runs"
    assert cfg.logs_dir == tmp_path / "logs"
    assert cfg.dataset_dir == tmp_path / "dataset"
    assert cfg.learned_exits_path == tmp_path / "learned_exits.json"
    assert cfg.video_dir == tmp_path / "videos"
    assert cfg.failed_video_dir == tmp_path / "videos" / "failed"


def test_default_data_root_is_project(tmp_path):
    assert Config().data_root == PROJECT_ROOT
    cfg, _ = config_from_dict({"data_dir": "sub"})
    assert cfg.data_root == PROJECT_ROOT / "sub"


def test_absolute_video_dir_kept(tmp_path):
    cfg, _ = config_from_dict({"data_dir": str(tmp_path), "recorder": {"video_dir": str(tmp_path / "v")}})
    assert cfg.video_dir == tmp_path / "v"


def test_nested_values_and_int_for_float():
    cfg, problems = config_from_dict({"recorder": {"fps": 3, "video_codec": "ffv1"},
                                      "training_frames": {"enabled": True, "max_mb": 10}})
    assert cfg.recorder.fps == 3.0 and isinstance(cfg.recorder.fps, float)
    assert cfg.recorder.video_codec == "FFV1"
    assert cfg.training_frames.enabled is True
    assert problems == []


def test_unknown_keys_reported():
    _, problems = config_from_dict({"bogus": 1, "analysis": {"collect_frames": True}})
    assert any("'bogus'" in p for p in problems)
    assert any("'analysis.collect_frames'" in p for p in problems)


@pytest.mark.parametrize("data", [
    {"monitor": "1"},
    {"monitor": 0},
    {"debug_snapshots": 1},
    {"recorder": {"fps": 0}},
    {"recorder": {"fps": 0.1}},
    {"recorder": {"fps": True}},
    {"recorder": {"video_codec": "MJPG"}},
    {"recorder": 5},
    {"log_level": "LOUD"},
    {"training_frames": {"jpeg_quality": 0}},
    {"analysis": {"map_area": [[0, 0], [1, 1]]}},
    {"analysis": {"map_area": [[0, 0], [1, 0], [2, 1]]}},
    {"analysis": {"map_area": [[0, 0], [1, 0], [0, 1, 0]]}},
    {"hotkeys": {"quit": "ctrl+alt+"}},
    {"hotkeys": {"quit": "ctrl+esc"}},
    {"hotkeys": {"record": "ctrl+"}},
    {"recorder": {"manual": True}, "hotkeys": {"record": "Ctrl+Alt+Q"}},   # the same key as quit
    {"hotkeys": {"pause": "ctrl+"}},
    {"hotkeys": {"pause": "ctrl+alt+q"}},                                  # the same key as quit
    {"hotkeys": {"pause": "ctrl+alt+r"}},                                  # the same key as record
    {"recorder": {"manual": "yes"}},
])
def test_bad_values_raise(data):
    with pytest.raises(ConfigError):
        config_from_dict(data)


def test_map_area_corners_of_the_screen_are_allowed():
    area = [[0, 0], [1, 0], [0.5, 1]]
    cfg, problems = config_from_dict({"analysis": {"map_area": area}})
    assert cfg.analysis.map_area == area and problems == []


def test_invalid_json(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{nope", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_example_config_is_valid():
    example = PROJECT_ROOT / "config.example.json"
    cfg, problems = config_from_dict(json.loads(example.read_text(encoding="utf-8")))
    assert problems == []


def test_settings_of_the_first_version_still_work():
    cfg, problems = config_from_dict({"log_level": "warn", "analysis": {
        "collect_frames": True, "collect_every": 2.0, "collect_max_mb": 500, "ocr_every": 1.0}})
    assert cfg.training_frames.enabled and cfg.training_frames.every == 2.0 and cfg.training_frames.max_mb == 500
    assert cfg.log_level == "WARNING"
    assert any("'training_frames.enabled'" in p for p in problems)            # told to rename it
    assert config_from_dict({"log_level": "CRITICAL"})[0].log_level == "CRITICAL"


def test_config_saved_with_a_byte_order_mark(tmp_path):
    # Notepad's "UTF-8 with BOM", and PowerShell 5.1's Set-Content -Encoding UTF8
    p = tmp_path / "config.json"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps({"monitor": 2}).encode("utf-8"))
    assert load_config(p)[0].monitor == 2


def test_old_setting_names_with_a_broken_new_section_raise_config_error():
    with pytest.raises(ConfigError):
        config_from_dict({"analysis": {"collect_frames": True}, "training_frames": 5})


def test_record_hotkey_may_share_a_key_when_recording_is_not_manual():
    cfg, _ = config_from_dict({"recorder": {"manual": False}, "hotkeys": {"record": "ctrl+alt+p"}})
    assert cfg.hotkeys.record == cfg.hotkeys.pause


def test_recording_is_started_by_hotkey_by_default():
    cfg, _ = config_from_dict({})
    assert cfg.recorder.manual is True and cfg.hotkeys.record == "ctrl+alt+r"
