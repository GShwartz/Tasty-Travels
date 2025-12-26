import threading
from queue import Queue

class CaptureThread(threading.Thread):
    def __init__(self, adb_ctrl):
        super().__init__(daemon=True)
        self.adb = adb_ctrl
        self.frame_queue = Queue(maxsize=1) # Keep only the freshest frame
        self.running = True

    def run(self):
        while self.running:
            frame = self.adb.get_screenshot()
            if frame is not None:
                if self.frame_queue.full():
                    self.frame_queue.get() # Drop old frame
                self.frame_queue.put(frame)

    def stop(self):
        self.running = False
        