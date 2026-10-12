# Uno Q LED matrix: the working path (research draft, 2026-10-04)

## UPDATE 2026-10-09 — the Docker relay is likely unnecessary
Arduino has since published an official standalone pip package,
`arduino-router-bridge` (PyPI, MPL-2.0,
github.com/arduino/arduino-router-bridge-py, sole dep msgpack).
It talks directly to the stock `arduino-router` systemd daemon on the Q
(`/usr/bin/arduino-router`, socket `/var/run/arduino-router.sock`,
MCU link `/dev/ttyHS1` @ 115200) — no App Lab / Docker needed on Linux.
Stock factory firmware registers ZERO Bridge methods (the boot logo is
local playback; `Reflash_Bootanimation.ino` contains no Bridge code), so a
custom sketch providing `draw` is still required to light the matrix — but
the Linux side collapses to `pip install arduino-router-bridge` + a few
lines of Python. Router built-ins only: `$/register`, `$/unregister`,
`$/reset`, `$/serial/open`, `$/serial/close`, `$/setMaxMsgSize`; no `$/list`.
Probing trick: error code 2 = method doesn't exist, 253 = exists but wrong
args — so candidate method names can be enumerated without guessing blind.
The TCP-relay architecture below is superseded; kept for reference.

## Why now
Paul said tonight (2026-10-04, after the fleet-watch session) he may do
development on the Arduino Q, and "if the AI agent doesn't work, it may be
better to do code that utilizes the small display." The dual repo's
`--display` flag for the built-in 8x13 blue LED matrix has been shelved
since 2026-09-27 because the direct-I2C driver doesn't work on real
hardware. This note establishes the actual working path, with draft code.

## The confirmed diagnosis: I2C is a dead end, by design
- The matrix is wired to the **STM32U585 MCU**, not the Linux SoC. There is
  no I2C device to poke from Linux: Paul's board scan found nothing at
  0x70 on any bus (memory/2026-09-27.md#L38). The old driver's
  community-documented 0x70 protocol described hardware that doesn't exist
  on this board.
- The only documented Linux->MCU path is **RouterBridge RPC** (internal
  SPI/remoteproc bridge). Multiple independent working projects use it for
  exactly this matrix.

## The working architecture
```
Linux (Debian)                        STM32U585 (Zephyr)
---------------                       ------------------
App Lab app container                 sketch.ino flashed to MCU
  python/main.py                        Arduino_RouterBridge: Bridge.provide("draw", draw)
    from arduino.app_utils import Bridge  Arduino_LED_Matrix: matrix.draw(104 row-major bytes, 0-7)
    Bridge.call("draw", frame_bytes)  --(msgpack-rpc over bridge)-->
  ^ listens on 127.0.0.1:29713
  | plain TCP: 104 bytes = one frame, b"CLEAR" = blank
ollama_duel.py --display (plain host script, no arduino SDK needed)
```

Key constraint (craignied/unoq-hello CLAUDE.md, verified): `arduino.app_utils`
is **not on host pip** -- it lives inside the App Lab app container image.
A plain `python3 ollama_duel.py` on the Q can never `import arduino.app_utils`
directly. Hence the relay split: the tiny App Lab app owns the Bridge, and
the duel script talks to it over localhost TCP.

## Draft files (in this directory, UNTESTED on hardware)
- `sketch.ino` -- STM32 side. `Arduino_RouterBridge` + `Arduino_LED_Matrix`,
  exposes `draw(std::vector<uint8_t>)`, paints directly in the provider
  (no shared framebuffer, so no mutex needed). Follows the linux.org "LED
  Matrix - Rain" walkthrough (complete verified example) and the
  mikersays/agent-arduino bridge skill.
- `sketch.yaml` -- build profile: `arduino:zephyr` platform, pins
  Arduino_RouterBridge 0.4.3 (as hsbl-ko-gyo/unoq-codex-matrix does).
- `app.yaml` -- DRAFT, minimal. Verify schema against the on-board example:
  `/var/lib/arduino-app-cli/examples/led-matrix-painter/app.yaml`.
- `main.py` -- the TCP relay. Syntax-checked (`python3 -m py_compile`);
  the `arduino.app_utils` import can only be exercised on the Q inside the
  app runtime.

## Feasibility checks (run on the Q)
```bash
which arduino-app-cli && arduino-app-cli --version
ls /var/lib/arduino-app-cli/examples/          # expect led-matrix-painter
cat /var/lib/arduino-app-cli/examples/led-matrix-painter/sketch/sketch.ino
python3 -c "from arduino.app_utils import Bridge"   # EXPECTED to fail on host: proves the SDK is container-only
```
If `arduino-app-cli` and the examples exist, the board is ready. Paul's Q
already has the full Arduino toolchain under /home/arduino/.arduino15
(~930M, found tonight), which is what builds/flashes the sketch -- his
decision to keep it just paid off.

## To try it
1. Copy this dir to the Q, e.g. `~/ArduinoApps/matrix-relay/` with
   `sketch/sketch.ino`, `sketch/sketch.yaml`, `python/main.py`, `app.yaml`.
2. `arduino-app-cli app start ~/ArduinoApps/matrix-relay/` (builds +
   flashes the sketch on first start; give it a minute).
3. `printf` a test frame: 104 bytes, e.g. all `\\x07` for full-on:
   `python3 -c "import socket; s=socket.create_connection(('127.0.0.1',29713)); s.sendall(bytes([7])*104); s.close()"`
4. Don't touch the matrix for ~20-30s after Linux boot (it shows the boot logo).

## Duel integration (proposed, not built)
Keep `UnoQMatrix`'s public interface (`DisplayUnavailable`, `show_text`,
`progress`, `clear`) and swap the I2C transport for a TCP client to the
relay. The pixel conversion (existing 13 column-bytes, LSB=top, to 104
row-major brightness bytes) is:
```python
import socket
def _tcp_write(columns):
    grid = bytearray(104)
    for c in range(13):
        b = columns[c] & 0x7F
        for r in range(8):
            grid[r * 13 + c] = 7 if (b >> r) & 1 else 0
    s = socket.create_connection(("127.0.0.1", 29713), timeout=2)
    try:
        s.sendall(bytes(grid))
    finally:
        s.close()
```
If the relay isn't running, the connect fails -> raise DisplayUnavailable ->
duel runs headless, same graceful fallback as today.

## Honest tradeoffs
- **Overhead**: the App Lab daemon + Docker + one small container on a 4GB
  box already running Ollama and duels. The relay container itself is tiny
  (python + socket server), but the daemon must be running.
- **Flash replaces factory firmware**: the sketch overwrites whatever the
  STM32 runs now (boot logo etc.). Reversible by reflashing, but say so.
- **It's cosmetic**: the matrix shows duel status (models, turn, tok/s) --
  nice, not load-bearing. If the relay ever flakes, `--display` degrades
  to headless with a warning.
- **Alternative**: skip the matrix entirely and keep duels headless.

## Sources
- linux.org thread "Arduino Uno Q - LED Matrix - Rain" (2026-10-02):
  https://www.linux.org/threads/arduino-uno-q-led-matrix-rain.71072/
  (complete sketch.ino + main.py, verified working by the author)
- craignied/unoq-hello CLAUDE.md: SDK container-only, 104-byte row-major
  frame format, on-board examples path
- mikersays/agent-arduino `.claude/skills/unoq-app-bridge/SKILL.md`:
  Bridge.provide/call pattern, provider-thread/mutex guidance, canonical
  `led-matrix-painter` example reference
- philippe86220/uno-q-webui-binary-clock-matrix: standalone Python service
  driving the matrix via Bridge RPC (`Bridge.call("updateTime", ...)`)
- himanshusaleria/air-drums: `msgpack-rpc -> Arduino router -> STM32 sketch`
  for the LED matrix; flash via `arduino:zephyr:unoq` core
- arduino/docs-content#2722: documents the App-Lab-only limitation and the
  lack of a standalone PyPI bridge library (why the relay split is needed)
- kanine/arduinoq-projects skill: matrix shows boot logo ~20-30s at startup;
  `Bridge.notify` for high-frequency updates
