import pygetwindow as gw
import logging

logger = logging.getLogger("TastyTravelsBot")

class WindowManager:
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
            