import pytest

from poe2layout.hotkey import MOD_NOREPEAT, VK_F1, parse_hotkey


def test_parse():
    assert parse_hotkey("ctrl+alt+q") == (MOD_NOREPEAT | 0x2 | 0x1, ord("Q"))
    assert parse_hotkey("Shift + F12") == (MOD_NOREPEAT | 0x4, VK_F1 + 11)
    assert parse_hotkey("win+5") == (MOD_NOREPEAT | 0x8, ord("5"))
    assert parse_hotkey("f1") == (MOD_NOREPEAT, VK_F1)
    assert parse_hotkey("f24") == (MOD_NOREPEAT, VK_F1 + 23)


@pytest.mark.parametrize("spec", ["ctrl+alt", "ctrl+f25", "ctrl+ä", "ctrl+esc", "", "q+f25", "q+f0", "q+esc"])
def test_parse_bad(spec):
    with pytest.raises(ValueError):
        parse_hotkey(spec)
