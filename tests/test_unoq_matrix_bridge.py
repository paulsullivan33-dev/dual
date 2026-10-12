"""Tests for BridgeMatrix -- the Router Bridge transport for the Uno Q's
8x13 LED matrix (see unoq_matrix.py).

A fake bridge stands in for arduino.router_bridge so the frame conversion
and failure handling are exercised with no hardware and no pip package.
Importing unoq_matrix must never require the package (lazy import, same as
smbus in the I2C driver).
"""

# pylint: disable=too-few-public-methods
# (test doubles below are single-purpose stand-ins)

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unoq_matrix
from unoq_matrix import BridgeMatrix, COLS, DisplayUnavailable


class FakeBridge:
    """Stand-in for arduino.router_bridge.Bridge: records draw calls."""

    def __init__(self, connect_ok=True, fail_call=False):
        self.frames = []
        self.connect_ok = connect_ok
        self.fail_call = fail_call

    def connect(self, timeout=None):
        return self.connect_ok

    def call(self, method, *args, timeout=None):
        if self.fail_call:
            raise RuntimeError("router exploded")
        assert method == "draw"
        self.frames.append(args[0])


def make_bridge_matrix(**kwargs):
    return BridgeMatrix(bridge=FakeBridge(**kwargs))


def test_import_needs_no_bridge_package():
    # If this module imported at all, the lazy-import design holds.
    assert "arduino.router_bridge" not in sys.modules
    assert "arduino" not in sys.modules


def test_init_clears_display():
    fb = FakeBridge()
    BridgeMatrix(bridge=fb)
    assert fb.frames == [bytes(8 * COLS)]  # one all-off probe frame


def test_no_package_raises_without_bridge_arg():
    # BridgeMatrix() with no injected bridge must try the real import,
    # which fails in this env -> DisplayUnavailable, not ImportError.
    try:
        BridgeMatrix()
    except DisplayUnavailable:
        return
    raise AssertionError("expected DisplayUnavailable")


def test_connect_failure_raises():
    try:
        make_bridge_matrix(connect_ok=False)
    except DisplayUnavailable:
        return
    raise AssertionError("expected DisplayUnavailable")


def test_call_failure_raises():
    try:
        make_bridge_matrix(fail_call=True)
    except DisplayUnavailable:
        return
    raise AssertionError("expected DisplayUnavailable")


def test_column_to_frame_conversion():
    # 'A' first glyph column is 0x7E = bits 1..6 -> rows 1..6 lit.
    fb = FakeBridge()
    m = BridgeMatrix(bridge=fb)
    fb.frames.clear()
    m.show_text("A")
    assert len(fb.frames) == 1
    frame = fb.frames[0]
    assert len(frame) == 8 * COLS
    assert frame[0 * COLS + 0] == 0  # row 0 dark
    assert frame[1 * COLS + 0] == 7  # rows 1..6 lit
    assert frame[6 * COLS + 0] == 7
    assert frame[7 * COLS + 0] == 0  # row 7 dark
    # second glyph column of 'A' is 0x11 = bits 0 and 4
    assert frame[0 * COLS + 1] == 7
    assert frame[4 * COLS + 1] == 7
    assert frame[1 * COLS + 1] == 0


def test_progress_bar_frame():
    fb = FakeBridge()
    m = BridgeMatrix(bridge=fb)
    fb.frames.clear()
    m.progress(1, 4)  # 25% of 13 columns -> 3 filled
    frame = fb.frames[-1]
    assert len(frame) == 8 * COLS
    # filled columns are 0x7F -> rows 0..6 lit, row 7 dark
    assert frame[0 * COLS + 0] == 7 and frame[6 * COLS + 2] == 7
    assert frame[7 * COLS + 0] == 0
    # column 3 onward dark
    assert frame[0 * COLS + 3] == 0 and frame[7 * COLS + 12] == 0


def test_mid_session_failure_raises():
    fb = FakeBridge()
    m = BridgeMatrix(bridge=fb)
    fb.fail_call = True
    try:
        m.clear()
    except DisplayUnavailable:
        return
    raise AssertionError("expected DisplayUnavailable")


def test_create_matrix_prefers_bridge():
    seen = []

    class FakeBM:
        def __init__(self):
            seen.append("bridge")

    orig_bm, orig_uq = unoq_matrix.BridgeMatrix, unoq_matrix.UnoQMatrix
    unoq_matrix.BridgeMatrix = FakeBM
    try:
        m = unoq_matrix.create_matrix()
        assert isinstance(m, FakeBM)
        assert seen == ["bridge"]
    finally:
        unoq_matrix.BridgeMatrix, unoq_matrix.UnoQMatrix = orig_bm, orig_uq


def test_create_matrix_falls_back_to_i2c():
    seen = []

    class FakeBM:
        def __init__(self):
            seen.append("bridge")
            raise DisplayUnavailable("nope")

    class FakeUQ:
        def __init__(self):
            seen.append("i2c")

    orig_bm, orig_uq = unoq_matrix.BridgeMatrix, unoq_matrix.UnoQMatrix
    unoq_matrix.BridgeMatrix, unoq_matrix.UnoQMatrix = FakeBM, FakeUQ
    try:
        m = unoq_matrix.create_matrix()
        assert isinstance(m, FakeUQ)
        assert seen == ["bridge", "i2c"]
    finally:
        unoq_matrix.BridgeMatrix, unoq_matrix.UnoQMatrix = orig_bm, orig_uq


def test_create_matrix_raises_when_nothing_works():
    class FailBM:
        def __init__(self):
            raise DisplayUnavailable("no bridge")

    class FailUQ:
        def __init__(self):
            raise DisplayUnavailable("no i2c")

    orig_bm, orig_uq = unoq_matrix.BridgeMatrix, unoq_matrix.UnoQMatrix
    unoq_matrix.BridgeMatrix, unoq_matrix.UnoQMatrix = FailBM, FailUQ
    try:
        unoq_matrix.create_matrix()
    except DisplayUnavailable as e:
        assert "no bridge" in str(e)  # the actionable error survives
        return
    finally:
        unoq_matrix.BridgeMatrix, unoq_matrix.UnoQMatrix = orig_bm, orig_uq
    raise AssertionError("expected DisplayUnavailable")
