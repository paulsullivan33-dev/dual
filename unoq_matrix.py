"""unoq_matrix.py -- driver for the Arduino Uno Q's built-in 8x13 LED matrix.

The Uno Q carries a monochrome blue 8x13 LED matrix (104 pixels) driven by
the onboard STM32U585 MCU. From the Linux (MPU) side it is reachable over
I2C at address 0x70: writing 13 column bytes (LSB = top pixel) sets the
frame. This follows the community-documented protocol; it has not yet been
verified against real hardware.

The driver is fully optional by design:

* ``smbus`` is imported lazily, inside the constructor, so importing this
  module is always safe -- even on machines with no I2C hardware at all.
* Every failure mode (missing smbus, missing bus, missing device, failed
  write) raises DisplayUnavailable. Callers should catch it, warn, and run
  headless.

Note: the matrix shows the boot logo for ~20-30 seconds during Linux
startup; only touch it after the board has finished booting.
"""

import time

I2C_ADDR = 0x70
COLS = 13
ROWS = 8

# 5x7 font, one entry per character: 5 column bytes, LSB = top pixel.
# Covers A-Z, 0-9, and the punctuation used for status readouts.
FONT_5X7 = {
    " ": (0x00, 0x00, 0x00, 0x00, 0x00),
    "0": (0x3E, 0x51, 0x49, 0x45, 0x3E),
    "1": (0x00, 0x42, 0x7F, 0x40, 0x00),
    "2": (0x42, 0x61, 0x51, 0x49, 0x46),
    "3": (0x21, 0x41, 0x45, 0x4B, 0x31),
    "4": (0x18, 0x14, 0x12, 0x7F, 0x10),
    "5": (0x27, 0x45, 0x45, 0x45, 0x39),
    "6": (0x3C, 0x4A, 0x49, 0x49, 0x30),
    "7": (0x01, 0x71, 0x09, 0x05, 0x03),
    "8": (0x36, 0x49, 0x49, 0x49, 0x36),
    "9": (0x06, 0x49, 0x49, 0x29, 0x1E),
    "A": (0x7E, 0x11, 0x11, 0x11, 0x7E),
    "B": (0x7F, 0x49, 0x49, 0x49, 0x36),
    "C": (0x3E, 0x41, 0x41, 0x41, 0x22),
    "D": (0x7F, 0x41, 0x41, 0x22, 0x1C),
    "E": (0x7F, 0x49, 0x49, 0x49, 0x41),
    "F": (0x7F, 0x48, 0x48, 0x48, 0x40),
    "G": (0x3E, 0x41, 0x49, 0x49, 0x7A),
    "H": (0x7F, 0x08, 0x08, 0x08, 0x7F),
    "I": (0x00, 0x41, 0x7F, 0x41, 0x00),
    "J": (0x20, 0x40, 0x41, 0x3F, 0x01),
    "K": (0x7F, 0x08, 0x14, 0x22, 0x41),
    "L": (0x7F, 0x40, 0x40, 0x40, 0x40),
    "M": (0x7F, 0x02, 0x0C, 0x02, 0x7F),
    "N": (0x7F, 0x04, 0x08, 0x10, 0x7F),
    "O": (0x3E, 0x41, 0x41, 0x41, 0x3E),
    "P": (0x7F, 0x09, 0x09, 0x09, 0x06),
    "Q": (0x3E, 0x41, 0x51, 0x21, 0x5E),
    "R": (0x7F, 0x09, 0x19, 0x29, 0x46),
    "S": (0x46, 0x49, 0x49, 0x49, 0x31),
    "T": (0x01, 0x01, 0x7F, 0x01, 0x01),
    "U": (0x7F, 0x40, 0x40, 0x40, 0x7F),
    "V": (0x1F, 0x20, 0x40, 0x20, 0x1F),
    "W": (0x3F, 0x40, 0x38, 0x40, 0x3F),
    "X": (0x63, 0x14, 0x08, 0x14, 0x63),
    "Y": (0x07, 0x08, 0x70, 0x08, 0x07),
    "Z": (0x61, 0x51, 0x49, 0x45, 0x43),
    ".": (0x00, 0x60, 0x60, 0x00, 0x00),
    ",": (0x00, 0x50, 0x30, 0x00, 0x00),
    ":": (0x00, 0x36, 0x36, 0x00, 0x00),
    "/": (0x20, 0x10, 0x08, 0x04, 0x02),
    "-": (0x08, 0x08, 0x08, 0x08, 0x08),
    "_": (0x40, 0x40, 0x40, 0x40, 0x40),
    "%": (0x62, 0x64, 0x08, 0x13, 0x23),
    "+": (0x08, 0x08, 0x3E, 0x08, 0x08),
    "!": (0x00, 0x00, 0x5F, 0x00, 0x00),
    "?": (0x02, 0x01, 0x51, 0x09, 0x06),
    "*": (0x14, 0x08, 0x3E, 0x08, 0x14),
}


class DisplayUnavailable(Exception):
    """The LED matrix can't be reached. Callers should catch this, warn,
    and continue without a display."""


class UnoQMatrix:
    """The Uno Q's built-in 8x13 LED matrix, via I2C from the Linux side.

    Pass ``bus_obj`` (anything with ``write_i2c_block_data``) to drive a
    fake in tests; otherwise smbus is imported lazily and the real bus is
    opened. Raises DisplayUnavailable on any failure.
    """

    def __init__(self, bus_number=1, address=I2C_ADDR, bus_obj=None):
        self.address = address
        if bus_obj is not None:
            self._bus = bus_obj
        else:
            try:
                import smbus
            except ImportError:
                try:
                    import smbus2 as smbus
                except ImportError:
                    smbus = None
            if smbus is None:
                raise DisplayUnavailable(
                    "python3-smbus is not installed; cannot reach the LED matrix"
                )
            try:
                self._bus = smbus.SMBus(bus_number)
            except OSError as e:
                raise DisplayUnavailable(f"cannot open I2C bus {bus_number}: {e}")
        # Probe the device with a clear; a missing device fails here.
        try:
            self.clear()
        except OSError as e:
            raise DisplayUnavailable(
                f"no LED matrix at I2C address 0x{address:02x}: {e}"
            )

    def _write(self, columns):
        """Write one 13-column frame; each value is a column, LSB = top."""
        frame = [(c & 0x7F) for c in list(columns)[:COLS]]
        frame += [0x00] * (COLS - len(frame))
        try:
            self._bus.write_i2c_block_data(self.address, 0x00, frame)
        except OSError as e:
            raise DisplayUnavailable(f"I2C write failed: {e}")

    def clear(self):
        """Blank the matrix."""
        self._write([0x00] * COLS)

    @staticmethod
    def render_text(text):
        """Render text to a list of column bytes (5px glyph + 1px spacing).
        Unknown characters fall back to '?'."""
        cols = []
        for ch in text.upper():
            cols.extend(FONT_5X7.get(ch, FONT_5X7["?"]))
            cols.append(0x00)
        if cols:
            cols.pop()  # no trailing space
        return cols

    def show_text(self, text, scroll_delay=0.1):
        """Show text: static if it fits in 13 columns, otherwise one scroll
        pass across the display."""
        cols = self.render_text(text)
        if len(cols) <= COLS:
            self._write(cols)
            return
        pad = [0x00] * COLS
        strip = pad + cols + pad
        for i in range(len(strip) - COLS + 1):
            self._write(strip[i:i + COLS])
            time.sleep(scroll_delay)

    def progress(self, done, total):
        """Show a horizontal progress bar: `done` of `total` steps."""
        total = max(1, total)
        filled = min(COLS, max(0, round(COLS * done / total)))
        self._write([0x7F] * filled + [0x00] * (COLS - filled))
