import subprocess
import cv2
import numpy as np
import time
import sys
from pathlib import Path
import os
import logging
import pygetwindow as gw


PROJECT_ROOT = Path(__file__).resolve().parent
ADB_BINARY = PROJECT_ROOT / "adb" / "adb.exe"
TEMPLATE_PATH = PROJECT_ROOT / "templates" / "energy.png"
MATCH_THRESHOLD = 0.88
FRAME_INTERVAL = 0.05 
COOLDOWN = 0.4

last_tap_time = 0.0
ADB_SERIAL = None 


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TastyTravelsBot")


def bring_window_to_front():
    """Finds the BlueStacks window and brings it to the foreground."""
    try:
        # Search for windows containing 'BlueStacks'
        windows = gw.getWindowsWithTitle('BlueStacks')
        if windows:
            win = windows[0]
            if win.isMinimized:
                win.restore()
            win.activate()
            logger.info("Focused window: %s", win.title)
        else:
            logger.warning("BlueStacks window not found. Please ensure the app is running.")
    except Exception as e:
        logger.error("Failed to focus window: %s", e)


def adb(cmd, serial=None):
    if not ADB_BINARY.exists():
        logger.error("ADB binary not found: %s", ADB_BINARY)
        sys.exit(1)

    full_cmd = [str(ADB_BINARY)]
    if serial:
        full_cmd += ["-s", serial]
    full_cmd += cmd

    result = subprocess.run(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        logger.error(result.stderr.decode(errors="ignore"))
        sys.exit(1)

    return result.stdout


def detect_bluestacks_device():
    output = adb(["devices"]).decode()
    devices = []
    for line in output.strip().splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        serial, state = parts
        if state == "device":
            devices.append(serial)

    for d in devices:
        if d.startswith("127.0.0.1"):
            logger.info("BlueStacks device detected: %s", d)
            return d

    if devices:
        logger.warning("No 127.0.0.1 device, using first available: %s", devices[0])
        return devices[0]

    logger.error("No active ADB devices found!")
    sys.exit(1)


def connect(serial):
    try:
        adb(["connect", serial])
        logger.info("Connected to ADB device %s", serial)
    except Exception as e:
        logger.error("Failed to connect to %s: %s", serial, e)
        sys.exit(1)


def close_adb_conn(serial):
    try:
        adb(["disconnect", serial])
        logger.info("Disconnected from ADB device %s", serial)
    except Exception as e:
        logger.warning("Failed to disconnect: %s", e)


def screencap(serial):
    raw = adb(["exec-out", "screencap", "-p"], serial)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        logger.error("Failed to decode screenshot")
        sys.exit(1)
    return img


def tap(x, y, serial):
    adb(["shell", "input", "tap", str(x), str(y)], serial)


def load_template():
    img = cv2.imread(str(TEMPLATE_PATH))
    if img is None:
        logger.error("Failed to load template: %s", TEMPLATE_PATH)
        sys.exit(1)
    return img


def find_and_act(frame, template, serial):
    global last_tap_time

    mask = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # TM_SQDIFF_NORMED or TM_CCORR_NORMED work best with masks
    result = cv2.matchTemplate(frame, template, cv2.TM_CCORR_NORMED, mask=mask)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    display_frame = frame.copy()

    # Note: Threshold might need adjustment when using masks
    if max_val >= 0.95: 
        h, w = template.shape[:2]
        
        # Draw the marking square
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(display_frame, top_left, bottom_right, (0, 255, 0), 3)
        cv2.putText(display_frame, f"ENERGY DETECTED: {max_val:.2f}", (top_left[0], top_left[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        now = time.time()
        if now - last_tap_time >= COOLDOWN:
            x = max_loc[0] + w // 2
            y = max_loc[1] + h // 2
            tap(x, y, serial)
            last_tap_time = now
            logger.info("TAP @ (%d,%d) | confidence=%.3f", x, y, max_val)

    cv2.imshow("Bot View (Energy Tracker)", display_frame)
    cv2.waitKey(1)


def main():
    global ADB_SERIAL

    logger.info("Using ADB binary: %s", ADB_BINARY)

    # Feature 1: Bring BlueStacks to front
    bring_window_to_front()

    # Feature 2: Connect to ADB
    ADB_SERIAL = detect_bluestacks_device()
    connect(ADB_SERIAL)

    template = load_template()
    logger.info("Template loaded: %s", TEMPLATE_PATH)

    logger.info("Starting real-time loop (Ctrl+C to exit)")
    try:
        while True:
            start = time.time()
            frame = screencap(ADB_SERIAL)
            
            # Feature 3: Find energy and mark it on screen
            find_and_act(frame, template, ADB_SERIAL)
            
            elapsed = time.time() - start
            delay = FRAME_INTERVAL - elapsed
            if delay > 0:
                time.sleep(delay)

    except KeyboardInterrupt:
        logger.info("Exiting bot...")

    finally:
        cv2.destroyAllWindows()
        close_adb_conn(ADB_SERIAL)


if __name__ == "__main__":
    main()