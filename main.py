import time
import os
import logging
from pathlib import Path
from Modules import WindowManager, ADBController, TemplateManager, TaskBot
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("TastyTravelsBot")

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent
    ADB_PATH = PROJECT_ROOT / "adb" / "adb.exe" if os.name == "nt" else PROJECT_ROOT / "adb" / "adb"
    
    # 1. Focus Emulator
    WindowManager.bring_to_front()
    
    # 2. Setup Device
    adb_ctrl = ADBController(ADB_PATH)
    adb_ctrl.setup_connection()
    
    # 3. Load Templates & Start Bot
    template_mgr = TemplateManager(PROJECT_ROOT / "templates")
    bot = TaskBot(adb_ctrl, template_mgr)

    logger.info("Bot Modules Loaded. Starting Tracker...")

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
