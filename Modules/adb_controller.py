import subprocess
import cv2
import numpy as np
import sys
import logging
from pathlib import Path

logger = logging.getLogger("TastyTravelsBot")

class ADBController:
    def __init__(self, adb_path):
        self.adb_binary = Path(adb_path)
        self.serial = None
        if not self.adb_binary.exists():
            print(f"ADB binary not found: {self.adb_binary}")
            sys.exit(1)

    def run_cmd(self, cmd):
        full_cmd = [str(self.adb_binary)]
        if self.serial: 
            full_cmd += ["-s", self.serial]
        full_cmd += cmd
        try:
            result = subprocess.run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.error("ADB Command timed out.")
            return b""

    def setup_connection(self):
        """Robustly scans for BlueStacks instances."""
        common_ports = ["5555", "5565", "5575", "5605"]
        logger.info("Probing BlueStacks ADB ports...")
        
        for port in common_ports:
            address = f"127.0.0.1:{port}"
            self.run_cmd(["connect", address])

        output = self.run_cmd(["devices"]).decode()
        lines = output.strip().splitlines()
        
        devices = []
        for line in lines:
            if "device" in line and not line.startswith("List"):
                parts = line.split()
                if len(parts) > 0:
                    devices.append(parts[0])

        if not devices:
            logger.critical("No ADB devices found! Please check BlueStacks Settings.")
            sys.exit(1)

        self.serial = next((d for d in devices if d.startswith("127.0.0.1")), devices[0])
        logger.info(f"Successfully bound to: {self.serial}")

    def get_screenshot(self):
        """Fast capture with PNG header validation."""
        raw = self.run_cmd(["exec-out", "screencap", "-p"])
        if not raw or len(raw) < 10 or not raw.startswith(b'\x89PNG'):
            return None
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    def tap(self, x, y):
        self.run_cmd(["shell", "input", "tap", str(x), str(y)])