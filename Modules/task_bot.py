import cv2
import time
import logging
import pytesseract
import numpy as np
import os

# FIXED PATH
TESSERACT_PATH = r'G:\Apps\Tesseract\tesseract.exe'
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

logger = logging.getLogger("TastyTravelsBot")

class TaskBot:
    def __init__(self, adb_ctrl, template_mgr):
        self.adb = adb_ctrl
        self.tm = template_mgr
        
        # Detection Thresholds
        self.threshold = 0.85      
        self.energy_thresh = 0.70  
        self.go_thresh = 0.60  # Raised - we need a better match
        
        self.mark_thickness = 3
        self.scale_percent = 0.6 

        self.grid_top_y_pct = 0.38
        self.grid_bottom_y_pct = 0.88
        self.header_height_pct = 0.35

    def _is_board_full(self, frame):
        h, w = frame.shape[:2]
        y_start = int(h * self.grid_top_y_pct)
        y_end = int(h * self.grid_bottom_y_pct)
        roi = frame[y_start:y_end, 0:w]
        if roi.size == 0: return False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        density = np.count_nonzero(edges) / (roi.shape[0] * roi.shape[1])
        return density > 0.18

    def _detect_go_button_by_color(self, frame):
        """Detect the green GO button using color detection"""
        h, w = frame.shape[:2]
        
        # Define the search region - ONLY the top area where GO button appears
        # GO button is between 10-25% from top
        search_top = int(h * 0.10)
        search_bottom = int(h * 0.30)  # Reduced to avoid game board
        search_roi = frame[search_top:search_bottom, 0:w]
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(search_roi, cv2.COLOR_BGR2HSV)
        
        # Bright green for GO button - more specific range
        lower_green = np.array([45, 120, 120])
        upper_green = np.array([75, 255, 255])
        
        # Create mask for green pixels
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for GO button - it should be relatively large and centered
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 1500 < area < 8000:  # GO button size range
                x, y, w_btn, h_btn = cv2.boundingRect(cnt)
                
                # Check aspect ratio (GO button is wider than tall)
                aspect_ratio = w_btn / h_btn if h_btn > 0 else 0
                if 1.8 < aspect_ratio < 3.5:
                    # Check if it's relatively centered (GO button appears center-ish)
                    center_x = x + w_btn / 2
                    if 0.3 * w < center_x < 0.7 * w:  # Center 40% of screen
                        candidates.append((x, y + search_top, w_btn, h_btn, area))
        
        # Return the largest candidate (GO button should be biggest green element in that area)
        if candidates:
            candidates.sort(key=lambda c: c[4], reverse=True)
            return candidates[0][:4]
        
        return None

    def process_frame(self, frame):
        if frame is None: return
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        
        board_clogged = self._is_board_full(frame)
        go_button_found = False

        # 1. ENERGY (GREEN)
        if self.tm.energy_template is not None:
            res_e = cv2.matchTemplate(frame, self.tm.energy_template, cv2.TM_CCOEFF_NORMED)
            _, max_val_e, _, max_loc_e = cv2.minMaxLoc(res_e)
            if max_val_e >= self.energy_thresh:
                eh, ew = self.tm.energy_template.shape[:2]
                cv2.rectangle(display_frame, max_loc_e, (max_loc_e[0]+ew, max_loc_e[1]+eh), (0, 255, 0), self.mark_thickness)

        # 2. GO BUTTON - COLOR-BASED DETECTION
        go_result = self._detect_go_button_by_color(frame)
        
        if go_result is not None:
            go_button_found = True
            x, y, w_btn, h_btn = go_result
            
            # Add padding to wrap around the entire button visually
            padding = 5
            x_padded = max(0, x - padding)
            y_padded = max(0, y - padding)
            w_padded = min(w, x + w_btn + padding) - x_padded
            h_padded = min(h, y + h_btn + padding) - y_padded
            
            # Draw rectangle around GO button (wrapped around it)
            color = (0, 0, 255)
            cv2.rectangle(display_frame, (x_padded, y_padded), (x_padded + w_padded, y_padded + h_padded), color, 2)
            
            # Removed center marker as requested
            
            logger.debug(f"GO button found at ({x}, {y}) size={w_btn}x{h_btn}")

        # 3. ITEMS (Generators & Orders)
        for category in ['generators', 'orders']:
            for item in self.tm.categories[category]:
                res_i = cv2.matchTemplate(frame, item['img'], cv2.TM_CCORR_NORMED, mask=item['mask'])
                _, max_val_i, _, max_loc_i = cv2.minMaxLoc(res_i)
                if max_val_i >= self.threshold:
                    ih, iw = item['img'].shape[:2]
                    color = (0, 255, 0) if category == 'generators' else (0, 165, 255)
                    cv2.rectangle(display_frame, max_loc_i, (max_loc_i[0]+iw, max_loc_i[1]+ih), color, self.mark_thickness)

        # UI OVERLAY
        status_text = "FULL" if board_clogged else "READY"
        cv2.putText(display_frame, status_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Removed GO DETECTED text as requested
        
        cv2.imshow("Bot Tracker", cv2.resize(display_frame, None, fx=self.scale_percent, fy=self.scale_percent))
        cv2.waitKey(1)