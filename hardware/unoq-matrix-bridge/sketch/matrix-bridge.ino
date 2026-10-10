// matrix-bridge.ino -- drive the Uno Q's 8x13 LED matrix from Linux.
//
// The matrix is wired to the STM32U585 MCU, not to the Linux SoC, so Linux
// can't touch it directly. This sketch exposes one RouterBridge RPC:
//
//   draw(<104 bytes>)  -- 8 rows x 13 cols, row-major, brightness 0-7
//
// Linux side needs only: pip install arduino-router-bridge
//   from arduino.router_bridge import Bridge
//   b = Bridge(); b.connect(); b.call("draw", bytes(104))
//
// Verified against the canonical pattern (linux.org "Plasma" walkthrough,
// Arduino's led-matrix-painter example) and the real Arduino_LED_Matrix
// header in ArduinoCore-zephyr: matrix.begin(), setGrayscaleBits(3),
// Bridge.begin(), Bridge.provide("draw", ...) with a
// void draw(std::vector<uint8_t>) provider. The provider paints directly --
// no shared framebuffer, so no mutex needed (loop() never touches hardware).
//
// Flash from the Q itself (no USB cable needed):
//   arduino-cli compile --fqbn arduino:zephyr:unoq sketch/
//   arduino-cli upload --fqbn arduino:zephyr:unoq:flash_mode=flash \
//       --port /dev/ttyHS1 sketch/
// If the port is busy, the router daemon holds it:
//   sudo systemctl stop arduino-router   # before flashing
//   sudo systemctl start arduino-router  # after, so the RPC socket is live
//
// Libraries (see sketch.yaml): Arduino_RouterBridge, Arduino_LED_Matrix
// Board core: arduino:zephyr (Uno Q)

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <vector>

Arduino_LED_Matrix matrix;

constexpr uint8_t ROWS = 8;
constexpr uint8_t COLS = 13;
constexpr size_t FRAME_SIZE = ROWS * COLS;  // 104

void draw(std::vector<uint8_t> frame) {
  if (frame.size() < FRAME_SIZE) {
    return;  // ignore short frames; keep the last good one on screen
  }
  for (size_t i = 0; i < FRAME_SIZE; i++) {
    if (frame[i] > 7) frame[i] = 7;  // clamp to 3-bit grayscale
  }
  matrix.draw(frame.data());
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);  // brightness levels 0-7 per pixel
  matrix.clear();
  Bridge.begin();
  Bridge.provide("draw", draw);
}

void loop() {
  delay(100);  // all work happens in the "draw" provider thread
}
