"""Tests for unoq_matrix.py -- the Arduino Uno Q's built-in 8x13 LED matrix.

A fake I2C bus stands in for smbus so the whole driver is exercised with
no hardware present. Importing unoq_matrix must never require smbus.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unoq_matrix
from unoq_matrix import COLS, DisplayUnavailable, UnoQMatrix


class FakeBus:
    """Stand-in for smbus.SMBus: records frames, optionally fails."""

    def __init__(self, fail=False):
        self.writes = []
        self.fail = fail

    def write_i2c_block_data(self, addr, reg, data):
        if self.fail:
            raise OSError("no such device")
        self.writes.append((addr, reg, list(data)))

    @property
    def last_frame(self):
        return self.writes[-1][2]


def make_matrix(fail=False):
    return UnoQMatrix(bus_obj=FakeBus(fail=fail))


def test_import_needs_no_smbus():
    # If this module imported at all, the lazy-import design holds: the
    # test env has no smbus installed.
    assert "smbus" not in sys.modules


def test_init_clears_display():
    bus = FakeBus()
    UnoQMatrix(bus_obj=bus)
    assert bus.last_frame == [0x00] * COLS


def test_missing_device_raises():
    try:
        make_matrix(fail=True)
    except DisplayUnavailable:
        return
    raise AssertionError("expected DisplayUnavailable")


def test_render_text_uses_font():
    assert UnoQMatrix.render_text("A") == [0x7E, 0x11, 0x11, 0x11, 0x7E]
    # glyph + 1px spacing, no trailing space: 5 + 1 + 5 = 11 columns
    assert len(UnoQMatrix.render_text("HI")) == 11


def test_unknown_char_falls_back_to_question_mark():
    assert UnoQMatrix.render_text("@") == UnoQMatrix.render_text("?")


def test_show_text_static_when_it_fits():
    bus = FakeBus()
    m = UnoQMatrix(bus_obj=bus)
    bus.writes.clear()
    m.show_text("HI")
    assert len(bus.writes) == 1  # one frame, no scrolling
    assert bus.last_frame == UnoQMatrix.render_text("HI") + [0x00] * (COLS - 11)


def test_show_text_scrolls_when_too_long():
    import unittest.mock as mock

    bus = FakeBus()
    m = UnoQMatrix(bus_obj=bus)
    bus.writes.clear()
    with mock.patch.object(unoq_matrix.time, "sleep", lambda s: None):
        m.show_text("TURN 12 OF 24")
    assert len(bus.writes) > 1
    first = bus.writes[0][2]
    last = bus.writes[-1][2]
    assert first[0] == 0x00 and last[-1] == 0x00  # padded scroll in/out


def test_progress_bar_math():
    bus = FakeBus()
    m = UnoQMatrix(bus_obj=bus)
    bus.writes.clear()
    m.progress(1, 4)  # 25% of 13 columns -> 3 filled
    frame = bus.last_frame
    assert frame[:3] == [0x7F] * 3
    assert frame[3:] == [0x00] * (COLS - 3)
    m.progress(4, 4)
    assert bus.last_frame == [0x7F] * COLS
    m.progress(0, 4)
    assert bus.last_frame == [0x00] * COLS


def test_clear():
    bus = FakeBus()
    m = UnoQMatrix(bus_obj=bus)
    bus.writes.clear()
    m.show_text("HI")
    m.clear()
    assert bus.last_frame == [0x00] * COLS


def test_write_failure_mid_session_raises():
    bus = FakeBus()
    m = UnoQMatrix(bus_obj=bus)
    bus.fail = True
    try:
        m.clear()
    except DisplayUnavailable:
        return
    raise AssertionError("expected DisplayUnavailable")
