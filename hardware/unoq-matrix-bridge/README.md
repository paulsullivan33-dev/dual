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

Run these commands from `hardware/unoq-matrix-bridge/` on the Q, with
Arduino CLI, the Arduino Zephyr board core, and the `Arduino_RouterBridge`
library installed. `matrix-bridge/sketch.yaml` records the library version;
`Arduino_LED_Matrix` is bundled with the board core.

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

An `RpcError ... code 2` saying no client provides `draw` means the service
is reachable but the sketch's RPC is unavailable. Confirm the companion
sketch is flashed and running. A connection failure instead points to the
`arduino-router` service or its socket at `/var/run/arduino-router.sock`.

## Use it with Dual

Display support is already integrated through `unoq_matrix.create_matrix()`.
After the smoke test succeeds, run from the repository root on the Q:

```bash
python3 ollama_duel.py scenarios/duel-example.json --profile arduino_q --display
```

Prepare the profile's models on the selected Ollama server before running.
You can also set `"display": true` in a scenario or profile; `--no-display`
disables it for one run. The display shows the turn number and generation
speed. The thermal guard can also show `COOLING` while a duel is paused.

The driver tries Router Bridge first and retains the old I2C driver as a
fallback. The I2C path did not work on the tested Uno Q hardware; installing
`python3-smbus` alone is not sufficient. If neither transport works, Dual
prints a warning and continues without the display. A display failure
during a duel also leaves the conversation running.

## Notes

- The matrix shows the boot logo for ~20-30s after power-on; leave it alone
  until Linux is up.
- Flashing replaces the factory STM32 firmware (reversible by reflashing).
- Display control is available through `ollama_duel.py`; `ollama_chat.py`
  does not expose a `--display` option.
