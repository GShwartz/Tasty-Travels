import subprocess
import cv2
import numpy as np
import sys
from pathlib import Path

class ADBController:
    def __init__(self, adb_path):
        self.adb_binary = Path(adb_path)
        self.serial = None
        if not self.adb_binary.exists():
            print(f"ADB binary not found: {self.adb_binary}")
            sys.exit(1)

    def run_cmd(self, cmd):
        full_cmd = [str(self.adb_binary)]
        if self.serial: full_cmd += ["-s", self.serial]
        full_cmd += cmd
        result = subprocess.run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout

    def setup_connection(self):
        output = self.run_cmd(["devices"]).decode()
        devices = [line.split()[0] for line in output.strip().splitlines() if "device" in line and not line.startswith("List")]
        self.serial = next((d for d in devices if d.startswith("127.0.0.1")), devices[0] if devices else None)
        if not self.serial:
            print("No ADB devices found!")
            sys.exit(1)
        self.run_cmd(["connect", self.serial])

    def get_screenshot(self):
        # exec-out is faster but can sometimes return empty buffers if the connection is unstable
        raw = self.run_cmd(["exec-out", "screencap", "-p"])
        
        # FIX: Check if we actually got image data (PNGs have a header)
        if not raw or len(raw) < 100:
            return None
            
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return img

    def tap(self, x, y):
        self.run_cmd(["shell", "input", "tap", str(x), str(y)])
        