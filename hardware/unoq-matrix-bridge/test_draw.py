#!/usr/bin/env python3
"""Smoke test for the Uno Q LED-matrix bridge sketch.

Flash the sketch first (see README.md), then on the Q:
    pip install arduino-router-bridge
    python3 test_draw.py

Lights the whole matrix, waits two seconds, then clears it.
If the sketch isn't flashed you'll get an RpcError "no client provides
the method" (code 2) -- that's the expected diagnostic, not a bug here.
"""

import sys
import time

try:
    from arduino.router_bridge import Bridge, RpcError
except ImportError:
    sys.exit("need the client library first:  pip install arduino-router-bridge")

ROWS, COLS = 8, 13
N = ROWS * COLS  # 104


def main() -> None:
    bridge = Bridge()  # default: unix:///var/run/arduino-router.sock
    if not bridge.connect(timeout=5):
        sys.exit("no connection to the Arduino router -- is arduino-router running?")
    try:
        print("all LEDs on...")
        bridge.call("draw", bytes([7]) * N, timeout=5)
        time.sleep(2)
        print("clearing...")
        bridge.call("draw", bytes(N), timeout=5)
    except RpcError as e:
        sys.exit(f"RPC error {e.code}: {e} -- sketch probably not flashed yet")
    finally:
        bridge.disconnect()
    print("done -- matrix should be dark again")


if __name__ == "__main__":
    main()
