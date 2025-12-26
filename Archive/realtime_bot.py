import subprocess
import cv2
import numpy as np
import time
import sys
from pathlib import Path
import os
import logging
import pygetwindow as gw

# -----------------------------
# LOGGER CONFIGURATION
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TastyTravelsBot")

# -----------------------------
# CLASSES
# -----------------------------

class WindowManager:
    """Handles window focus and activation."""
    @staticmethod
    def bring_to_front(title_keyword='BlueStacks'):
        try:
            windows = gw.getWindowsWithTitle(title_keyword)
            if windows:
                win = windows[0]
                if win.isMinimized: win.restore()
                win.activate()
                logger.info(f"Focused window: {win.title}")
            else:
                logger.warning(f"{title_keyword} window not found.")
        except Exception as e:
            logger.error(f"Failed to focus window: {e}")

class ADBController:
    """Handles device communication: screenshots and input."""
    def __init__(self, adb_path):
        self.adb_binary = Path(adb_path)
        self.serial = None
        if not self.adb_binary.exists():
            logger.error(f"ADB binary not found: {self.adb_binary}")
            sys.exit(1)

    def run_cmd(self, cmd):
        full_cmd = [str(self.adb_binary)]
        if self.serial: full_cmd += ["-s", self.serial]
        full_cmd += cmd
        result = subprocess.run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout

    def setup_connection(self):
        output = self.run_cmd(["devices"]).decode()
        devices = [line.split()[0] for line in output.strip().splitlines()[1:] if "device" in line]
        self.serial = next((d for d in devices if d.startswith("127.0.0.1")), devices[0] if devices else None)
        if not self.serial:
            logger.error("No ADB devices found!")
            sys.exit(1)
        self.run_cmd(["connect", self.serial])
        logger.info(f"Connected to {self.serial}")

    def get_screenshot(self):
        raw = self.run_cmd(["exec-out", "screencap", "-p"])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return img

    def tap(self, x, y):
        """Tapping logic kept aside as requested."""
        self.run_cmd(["shell", "input", "tap", str(x), str(y)])

class TemplateManager:
    """Handles loading and masking of images from all template directories."""
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.categories = {'generators': [], 'orders': []}
        self.energy_template = None
        self.energy_mask = None
        self.load_all()

    def _create_mask(self, template):
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        return mask

    def load_all(self):
        # 1. Load standalone Energy template
        energy_path = self.base_path / "energy.png"
        if energy_path.exists():
            self.energy_template = cv2.imread(str(energy_path))
            self.energy_mask = self._create_mask(self.energy_template)
            logger.info("Energy template loaded.")

        # 2. Load Generators and Orders
        for category in ['generators', 'orders']:
            comp_path = self.base_path / category / "components"
            comp_path.mkdir(parents=True, exist_ok=True)
            for img_path in comp_path.glob("*.png"):
                img = cv2.imread(str(img_path))
                if img is not None:
                    self.categories[category].append({
                        'id': img_path.stem,
                        'img': img,
                        'mask': self._create_mask(img)
                    })
        logger.info(f"Loaded {len(self.categories['generators'])} generators and {len(self.categories['orders'])} orders.")

class TaskBot:
    def __init__(self, adb_ctrl, template_mgr):
        self.adb = adb_ctrl
        self.tm = template_mgr
        self.threshold = 0.88
        self.mark_thickness = 1 # Thinner marks as requested

    def process_frame(self, frame):
        display_frame = frame.copy()
        found_gen_ids = []

        # --- ENERGY TRACKING (Mark Green) ---
        if self.tm.energy_template is not None:
            res = cv2.matchTemplate(frame, self.tm.energy_template, cv2.TM_CCORR_NORMED, mask=self.tm.energy_mask)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= 0.95:
                h, w = self.tm.energy_template.shape[:2]
                cv2.rectangle(display_frame, max_loc, (max_loc[0]+w, max_loc[1]+h), (0, 255, 0), self.mark_thickness)

        # --- GENERATORS TRACKING (Mark Green) ---
        for item in self.tm.categories['generators']:
            res = cv2.matchTemplate(frame, item['img'], cv2.TM_CCORR_NORMED, mask=item['mask'])
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= self.threshold:
                found_gen_ids.append(item['id'])
                h, w = item['img'].shape[:2]
                cv2.rectangle(display_frame, max_loc, (max_loc[0]+w, max_loc[1]+h), (0, 255, 0), self.mark_thickness)

        # --- ORDERS TRACKING (Mark Blue if Met, else Orange) ---
        for item in self.tm.categories['orders']:
            res = cv2.matchTemplate(frame, item['img'], cv2.TM_CCORR_NORMED, mask=item['mask'])
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= self.threshold:
                h, w = item['img'].shape[:2]
                is_met = item['id'] in found_gen_ids
                # Blue (255,0,0) if met, Orange (0,165,255) if not
                color = (255, 0, 0) if is_met else (0, 165, 255)
                cv2.rectangle(display_frame, max_loc, (max_loc[0]+w, max_loc[1]+h), color, self.mark_thickness)
                
        cv2.imshow("Bot Tracker", display_frame)
        cv2.waitKey(1)

# -----------------------------
# MAIN
# -----------------------------
def main():
    PROJECT_ROOT = Path(__file__).resolve().parent
    ADB_PATH = PROJECT_ROOT / "adb" / "adb.exe" if os.name == "nt" else PROJECT_ROOT / "adb" / "adb"
    
    WindowManager.bring_to_front()
    adb_ctrl = ADBController(ADB_PATH)
    adb_ctrl.setup_connection()
    
    template_mgr = TemplateManager(PROJECT_ROOT / "templates")
    bot = TaskBot(adb_ctrl, template_mgr)

    try:
        while True:
            frame = adb_ctrl.get_screenshot()
            if frame is not None:
                bot.process_frame(frame)
            time.sleep(0.05)
    except KeyboardInterrupt:
        logger.info("Exiting...")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    