import cv2
import time
import logging
import pytesseract
import numpy as np
import os

# FIX: Pointed to your specific installation path
TESSERACT_PATH = r'G:\Apps\Tesseract\tesseract.exe'

# Safety check to confirm the .exe is actually there
if not os.path.exists(TESSERACT_PATH):
    print(f"\n[ERROR] Tesseract NOT found at: {TESSERACT_PATH}")
    print("Check if 'tesseract.exe' exists inside G:\\Apps\\Tesseract\\\n")
else:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

logger = logging.getLogger("TastyTravelsBot")

class TaskBot:
    def __init__(self, adb_ctrl, template_mgr):
        self.adb = adb_ctrl
        self.tm = template_mgr
        self.threshold = 0.88
        self.mark_thickness = 2
        self.scale_percent = 0.6 
        self.last_tap_time = 0
        self.wait_duration = 3.0 

    def _ocr_validate_go(self, frame, x, y, w, h):
        """Extracts text and validates to stop misfires on green backgrounds."""
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0: return False
        
        # Pre-process for OCR: Create high-contrast black/white image
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        try:
            # Use PSM 8 (Single word mode) for better accuracy on buttons
            text = pytesseract.image_to_string(thresh, config='--psm 8').strip().upper()
            
            # Match 'GO' or fragments of it to account for OCR noise
            is_match = "GO" in text or (len(text) >= 2 and "G" in text)
            
            if is_match:
                logger.info(f"OCR VERIFIED 'GO' TEXT: {text}")
            return is_match
        except Exception as e:
            logger.error(f"OCR Execution failed: {e}")
            return False

    def tap_center(self, x, y, w, h):
        center_x = int(x + (w / 2))
        center_y = int(y + (h / 2))
        self.adb.tap(center_x, center_y)
        self.last_tap_time = time.time()

    def process_frame(self, frame):
        if frame is None: return
        display_frame = frame.copy()
        found_gen_ids = []
        current_time = time.time()

        # 1. Energy Tracking (Green Rectangle)
        if self.tm.energy_template is not None:
            res = cv2.matchTemplate(frame, self.tm.energy_template, cv2.TM_CCORR_NORMED, mask=self.tm.energy_mask)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= 0.95:
                h, w = self.tm.energy_template.shape[:2]
                cv2.rectangle(display_frame, max_loc, (max_loc[0]+w, max_loc[1]+h), (0, 255, 0), self.mark_thickness)

        # 2. GO Button Tracking (Red Rectangle + OCR)
        if self.tm.go_template is not None:
            res_go = cv2.matchTemplate(frame, self.tm.go_template, cv2.TM_CCORR_NORMED, mask=self.tm.go_mask)
            _, max_val_go, _, max_loc_go = cv2.minMaxLoc(res_go)
            
            # High threshold to ignore the 'ghost' matches in the top-left
            if max_val_go >= 0.82: 
                h, w = self.tm.go_template.shape[:2]
                
                # Use OCR to confirm it's a real button, not just a green object
                if self._ocr_validate_go(frame, max_loc_go[0], max_loc_go[1], w, h):
                    cv2.rectangle(display_frame, max_loc_go, (max_loc_go[0]+w, max_loc_go[1]+h), (0, 0, 255), self.mark_thickness)

                    if current_time - self.last_tap_time > self.wait_duration:
                        self.tap_center(max_loc_go[0], max_loc_go[1], w, h)

        # 3. Generators & Orders (Green/Orange Rectangles)
        for category in ['generators', 'orders']:
            for item in self.tm.categories[category]:
                res = cv2.matchTemplate(frame, item['img'], cv2.TM_CCORR_NORMED, mask=item['mask'])
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val >= self.threshold:
                    h, w = item['img'].shape[:2]
                    color = (0, 255, 0) if category == 'generators' else (0, 165, 255)
                    if category == 'generators': found_gen_ids.append(item['id'])
                    cv2.rectangle(display_frame, max_loc, (max_loc[0]+w, max_loc[1]+h), color, self.mark_thickness)

        # UI Display
        width = int(display_frame.shape[1] * self.scale_percent)
        height = int(display_frame.shape[0] * self.scale_percent)
        cv2.imshow("Bot Tracker", cv2.resize(display_frame, (width, height)))
        cv2.waitKey(1)
        