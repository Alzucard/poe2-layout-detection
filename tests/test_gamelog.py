from poe2layout.gamelog import GameLog, ZoneEvent, find_client_log
from poe2layout.zones import Instance

PREFIX = "2026/11/17 12:00:00 399237921 2caa22f9 "      # date, time, ms counter, thread id (like in Client.txt)


def lines(*ls: str, tag: str = "[INFO Client 39372]") -> str:
    return "".join(f"{PREFIX}{tag} {s}\n" for s in ls)


def enter(code: str, seed: int, name: str, level: int = 10) -> str:
    return (lines(f'Generating level {level} area "{code}" with seed {seed}', tag="[DEBUG Client 39372]")
            + lines(f"[SCENE] Set Source [{name}]"))


def test_first_poll_reports_only_the_current_zone(tmp_path):
    p = tmp_path / "Client.txt"
    p.write_text(enter("G1_1", 1, "The Riverbank") + enter("G1_2", 2, "Clearfell", 3), encoding="utf-8")
    log = GameLog(p)
    assert log.poll() == [ZoneEvent("Clearfell", Instance("G1_2", 2), 3)]
    assert log.poll() == []


def test_new_lines_and_special_scenes(tmp_path):
    p = tmp_path / "Client.txt"
    p.write_text("", encoding="utf-8")
    log = GameLog(p)
    assert log.poll() == []
    with open(p, "a", encoding="utf-8") as f:
        f.write(enter("G1_4", 7, "The Grelwood"))
        f.write(lines("[SCENE] Set Source [(null)]", "[SCENE] Set Source [Act 1]"))   # no Generating line
        f.write(lines("[SCENE] Set Source [(unknown)]"))
    evs = log.poll()
    assert evs == [ZoneEvent("The Grelwood", Instance("G1_4", 7), 10), ZoneEvent(None, None, None)]


def test_partial_line_is_finished_by_next_poll(tmp_path):
    p = tmp_path / "Client.txt"
    p.write_text("", encoding="utf-8")
    log = GameLog(p)
    log.poll()
    whole = enter("G1_2", 5, "Clearfell")
    cut = len(whole) - 8
    with open(p, "a", encoding="utf-8") as f:
        f.write(whole[:cut])
    assert log.poll() == []
    with open(p, "a", encoding="utf-8") as f:
        f.write(whole[cut:])
    assert log.poll() == [ZoneEvent("Clearfell", Instance("G1_2", 5), 10)]


def test_started_mid_file_skips_cut_line(tmp_path):
    p = tmp_path / "Client.txt"
    p.write_text("x" * 500 + "\n" + enter("G1_2", 5, "Clearfell"), encoding="utf-8")
    log = GameLog(p, tail_bytes=p.stat().st_size - 100)    # starts inside the junk line
    assert log.poll() == [ZoneEvent("Clearfell", Instance("G1_2", 5), 10)]


def test_every_zone_after_the_first_poll_is_reported(tmp_path):
    p = tmp_path / "Client.txt"
    p.write_text(enter("G1_1", 1, "The Riverbank"), encoding="utf-8")
    log = GameLog(p)
    log.poll()
    with open(p, "a", encoding="utf-8") as f:
        f.write(enter("G1_2", 2, "Clearfell") + enter("G1_3", 3, "Mud Burrow"))
    assert [e.zone for e in log.poll()] == ["Clearfell", "Mud Burrow"]


def test_login_screen_forgets_the_level_being_generated(tmp_path):
    p = tmp_path / "Client.txt"
    p.write_text("", encoding="utf-8")
    log = GameLog(p)
    log.poll()
    with open(p, "a", encoding="utf-8") as f:
        f.write(lines("[SCENE] Set Source [Act 1]"))                 # no Generating line seen yet
        f.write(lines('Generating level 5 area "G1_2" with seed 9', tag="[DEBUG Client 39372]"))
        f.write(lines("[SCENE] Set Source [(unknown)]", "[SCENE] Set Source [Act 1]"))
    assert log.poll() == [ZoneEvent(None, None, None)]


def test_log_cleared(tmp_path):
    p = tmp_path / "Client.txt"
    p.write_text(enter("G1_2", 5, "Clearfell") * 3, encoding="utf-8")
    log = GameLog(p)
    log.poll()
    p.write_text(enter("G1_4", 6, "The Grelwood"), encoding="utf-8")
    assert log.poll() == [ZoneEvent("The Grelwood", Instance("G1_4", 6), 10)]


def test_find_client_log_configured(tmp_path):
    p = tmp_path / "Client.txt"
    assert find_client_log(str(p)) is None
    p.write_text("", encoding="utf-8")
    assert find_client_log(str(p)) == p


def test_chat_cannot_fake_a_zone_change(tmp_path):
    p = tmp_path / "Client.txt"
    p.write_text(enter("G1_2", 5, "Clearfell"), encoding="utf-8")
    log = GameLog(p)
    log.poll()
    with open(p, "a", encoding="utf-8") as f:
        for who in ("#Troll: ", "@From Troll: ", "$Troll: ", "%Troll: ", "#Troll: [INFO Client 1] "):
            f.write(lines(f'{who}Generating level 1 area "G9_9" with seed 5', f"{who}[SCENE] Set Source [Fake]"))
        f.write(enter("G1_4", 6, "The Grelwood"))
    assert log.poll() == [ZoneEvent("The Grelwood", Instance("G1_4", 6), 10)]


def test_windows_line_endings(tmp_path):
    p = tmp_path / "Client.txt"
    p.write_bytes(enter("G1_2", 5, "Clearfell").replace("\n", "\r\n").encode())
    assert GameLog(p).poll() == [ZoneEvent("Clearfell", Instance("G1_2", 5), 10)]


def test_text_after_the_seed_is_allowed(tmp_path):
    p = tmp_path / "Client.txt"
    p.write_text(lines('Generating level 3 area "G1_2" with seed 5 (cached)', tag="[DEBUG Client 39372]")
                 + lines("[SCENE] Set Source [Clearfell]"), encoding="utf-8")
    assert GameLog(p).poll() == [ZoneEvent("Clearfell", Instance("G1_2", 5), 3)]
