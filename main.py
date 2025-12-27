from Modules import WindowManager, ADBController, TemplateManager, TaskBot, init_logger, CaptureThread
from pathlib import Path
import logging
import time
import os
import cv2

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("TastyTravelsBot")

def main():
    # Setup Paths
    PROJECT_ROOT = Path(__file__).resolve().parent
    log_file = PROJECT_ROOT / "tasty_travels_bot.log"
    
    # Initialize Logger
    main_logger = init_logger(log_file, "Main")

    # ADB Path Logic
    ADB_PATH = PROJECT_ROOT / "adb" / "adb.exe" if os.name == "nt" else PROJECT_ROOT / "adb" / "adb"

    # 1. Focus Emulator
    main_logger.info("Bringing emulator window to front...")
    WindowManager.bring_to_front()
    
    # 2. Setup Device
    main_logger.info("Setting up ADB connection...")
    adb_ctrl = ADBController(ADB_PATH)
    adb_ctrl.setup_connection()
    
    # 3. Load Templates
    main_logger.info("Loading templates...")
    template_mgr = TemplateManager(PROJECT_ROOT / "templates")
    
    # 4. Initialize Bot
    bot = TaskBot(adb_ctrl, template_mgr)
    main_logger.info("Bot Modules Loaded.")

    # 5. Start Capture Thread
    main_logger.info("Starting background capture thread...")
    capture = CaptureThread(adb_ctrl)
    capture.start()
    
    main_logger.info("Tracker started. Press Ctrl+C to exit.")

    try:
        while True:
            # Pull frame from the Thread Queue
            if not capture.frame_queue.empty():
                frame = capture.frame_queue.get()
                bot.process_frame(frame)
            
            # Control the loop frequency
            time.sleep(0.01)

    except KeyboardInterrupt:
        main_logger.info("Exiting...")
        
    finally:
        # Cleanup threads
        capture.stop()
        bot.cleanup()
        cv2.destroyAllWindows()
        main_logger.info("Cleanup complete.")

if __name__ == "__main__":
    main()
    