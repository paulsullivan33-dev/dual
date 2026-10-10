# Uno Q LED matrix bridge

Drive the Q's built-in 8x13 LED matrix from Linux over the Arduino Router Bridge.
No Docker, no App Lab — just a sketch on the STM32 and a pip package on Linux.

## Layout

- `matrix-bridge/` — STM32 sketch exposing one RPC: `draw(<104 bytes>)`,
  8 rows x 13 cols, row-major, brightness 0-7 per pixel.
- `test_draw.py` — Linux smoke test: lights all LEDs, waits, clears.
- `NOTES.md` — research background (why the matrix needs the MCU, the
  dead I2C path, the superseded Docker-relay design).

## Flash the sketch (on the Q)

```bash
ls -la /var/run/arduino-router.sock   # router daemon alive?
# if the upload below complains the port is busy:
sudo systemctl stop arduino-router

arduino-cli compile --fqbn arduino:zephyr:unoq matrix-bridge/
arduino-cli upload --fqbn arduino:zephyr:unoq:flash_mode=flash \
    --port /dev/ttyHS1 matrix-bridge/

sudo systemctl start arduino-router   # if you stopped it
```

`flash_mode=flash` puts it in non-volatile memory so it survives power cycles.

## Test it

```bash
pip install arduino-router-bridge
python3 test_draw.py
```

An `RpcError ... code 2` means the sketch isn't flashed yet (no provider
for `draw`) — that's the diagnostic, not a failure of the test.

## Notes

- The matrix shows the boot logo for ~20-30s after power-on; leave it alone
  until Linux is up.
- Flashing replaces the factory STM32 firmware (reversible by reflashing).
- Next step after the smoke test: point `ollama_duel.py --display` at
  `Bridge.call("draw", frame)` instead of the old I2C driver.
