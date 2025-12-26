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
        self.go_thresh = 0.60
        
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

    def _detect_all_go_buttons(self, frame):
        """Detect ALL green GO buttons using color detection - returns list of detections"""
        h, w = frame.shape[:2]
        
        # Define the search region - ONLY the header area where GO buttons appear
        # GO buttons are between 10-30% from top (the draggable header)
        search_top = int(h * 0.10)
        search_bottom = int(h * 0.30)
        search_roi = frame[search_top:search_bottom, 0:w]
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(search_roi, cv2.COLOR_BGR2HSV)
        
        # Bright green for GO button - specific range
        lower_green = np.array([45, 120, 120])
        upper_green = np.array([75, 255, 255])
        
        # Create mask for green pixels
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for ALL GO buttons
        go_buttons = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 1500 < area < 8000:  # GO button size range
                x, y, w_btn, h_btn = cv2.boundingRect(cnt)
                
                # Check aspect ratio (GO button is wider than tall)
                aspect_ratio = w_btn / h_btn if h_btn > 0 else 0
                if 1.8 < aspect_ratio < 3.5:
                    # Adjust y coordinate back to original frame coordinates
                    go_buttons.append({
                        'x': x,
                        'y': y + search_top,
                        'w': w_btn,
                        'h': h_btn,
                        'area': area
                    })
        
        # Sort by x coordinate (left to right) for consistent ordering
        go_buttons.sort(key=lambda btn: btn['x'])
        
        return go_buttons

    def process_frame(self, frame):
        if frame is None: return
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        
        board_clogged = self._is_board_full(frame)

        # 1. ENERGY (GREEN)
        if self.tm.energy_template is not None:
            res_e = cv2.matchTemplate(frame, self.tm.energy_template, cv2.TM_CCOEFF_NORMED)
            _, max_val_e, _, max_loc_e = cv2.minMaxLoc(res_e)
            if max_val_e >= self.energy_thresh:
                eh, ew = self.tm.energy_template.shape[:2]
                cv2.rectangle(display_frame, max_loc_e, (max_loc_e[0]+ew, max_loc_e[1]+eh), (0, 255, 0), self.mark_thickness)

        # 2. ALL GO BUTTONS - COLOR-BASED DETECTION
        go_buttons = self._detect_all_go_buttons(frame)
        
        if go_buttons:
            for idx, btn in enumerate(go_buttons):
                x, y = btn['x'], btn['y']
                w_btn, h_btn = btn['w'], btn['h']
                
                # Add padding to wrap around the entire button visually
                padding = 5
                x_padded = max(0, x - padding)
                y_padded = max(0, y - padding)
                w_padded = min(w, x + w_btn + padding) - x_padded
                h_padded = min(h, y + h_btn + padding) - y_padded
                
                # Draw rectangle around GO button (wrapped around it)
                color = (0, 0, 255)  # Red
                cv2.rectangle(display_frame, (x_padded, y_padded), 
                            (x_padded + w_padded, y_padded + h_padded), color, 2)
                
                # Add number label to each GO button
                label = f"GO{idx+1}"
                cv2.putText(display_frame, label, (x_padded, y_padded - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                logger.debug(f"GO button #{idx+1} at ({x}, {y}) size={w_btn}x{h_btn}")

        # 3. ITEMS (Generators & Orders)
        for category in ['generators', 'orders']:
            for item in self.tm.categories[category]:
                res_i = cv2.matchTemplate(frame, item['img'], cv2.TM_CCORR_NORMED, mask=item['mask'])
                _, max_val_i, _, max_loc_i = cv2.minMaxLoc(res_i)
                if max_val_i >= self.threshold:
                    ih, iw = item['img'].shape[:2]
                    color = (0, 255, 0) if category == 'generators' else (0, 165, 255)
                    cv2.rectangle(display_frame, max_loc_i, (max_loc_i[0]+iw, max_loc_i[1]+ih), color, self.mark_thickness)

        # UI OVERLAY - Show count of GO buttons detected
        status_text = f"GO Buttons: {len(go_buttons)}" if go_buttons else ""
        cv2.putText(display_frame, status_text, (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Bot Tracker", cv2.resize(display_frame, None, fx=self.scale_percent, fy=self.scale_percent))
        cv2.waitKey(1)
        