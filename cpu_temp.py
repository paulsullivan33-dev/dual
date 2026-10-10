#!/usr/bin/env python3
"""Print this machine's CPU temperature.

Reads /sys/class/thermal (thermal_zone0 is the CPU on Raspberry Pi and
most ARM boards, including the Arduino Uno Q). Run it on whichever box
you're wondering about -- handy over SSH when a duel is cooking:

    python3 cpu_temp.py
    ArduinoQ: CPU temp 54.2C
"""

import glob
import os
import socket
import sys


def read_temps():
    """Return {zone_temp_path: degrees_C} for every readable thermal zone."""
    temps = {}
    for path in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        try:
            with open(path, encoding="utf-8") as f:
                temps[path] = int(f.read().strip()) / 1000.0
        except (OSError, ValueError):
            continue
    return temps


def main():
    temps = read_temps()
    host = socket.gethostname()
    if not temps:
        print(f"{host}: no readable CPU temperature sensor found.",
              file=sys.stderr)
        return 1
    cpu_zone = min(temps)  # thermal_zone0 sorts first
    print(f"{host}: CPU temp {temps[cpu_zone]:.1f}C")
    for path, temp in temps.items():
        if path != cpu_zone:
            zone = os.path.basename(os.path.dirname(path))
            print(f"  {zone}: {temp:.1f}C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
