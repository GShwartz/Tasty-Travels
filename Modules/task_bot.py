import cv2
import time
import logging
import pytesseract
import numpy as np
import os
import threading
from queue import Queue
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_ROOT / "proc"
PROC_DIR.mkdir(parents=True, exist_ok=True)

TESSERACT_PATH = r'G:\Apps\Tesseract\tesseract.exe'
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

logger = logging.getLogger("TastyTravelsBot")

class OCRWorker(threading.Thread):
    """Background thread for OCR processing"""
    def __init__(self):
        super().__init__(daemon=True)
        self.input_queue = Queue(maxsize=1)
        self.result = None
        self.running = True
        self.debug_counter = 0  # For saving debug images
        
    def run(self):
        while self.running:
            try:
                # Get ROI from queue (blocks until available)
                roi = self.input_queue.get(timeout=0.1)
                if roi is None:
                    continue
                
                # Save first few ROIs for debugging
                if self.debug_counter < 3:
                    cv2.imwrite(str(PROC_DIR / f"debug_roi_{self.debug_counter}.png"), roi)
                
                # Convert to grayscale
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # Simple binary threshold - white text on transparent/colored background
                # The timer text is usually white/light colored
                _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                
                if self.debug_counter < 3:
                    cv2.imwrite(str(PROC_DIR / f"debug_binary_{self.debug_counter}.png"), binary)
                
                # Upscale significantly
                scale_factor = 6
                scaled = cv2.resize(binary, None, fx=scale_factor, fy=scale_factor, 
                                   interpolation=cv2.INTER_LINEAR)
                
                if self.debug_counter < 3:
                    cv2.imwrite(str(PROC_DIR / f"debug_scaled_{self.debug_counter}.png"), scaled)
                    self.debug_counter += 1
                
                # Multiple PSM modes to try
                configs = [
                    r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:',  # Single line
                    r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789:',  # Single word
                    r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789:', # Raw line
                ]
                
                best_result = None
                best_length = 0
                
                for config in configs:
                    try:
                        text = pytesseract.image_to_string(scaled, config=config).strip()
                        text = ''.join(c for c in text if c.isdigit() or c == ':')
                        
                        # Prefer results with colons and longer results
                        if ':' in text and len(text) > best_length:
                            best_result = text
                            best_length = len(text)

                        elif len(text) > best_length and best_result is None:
                            best_result = text
                            best_length = len(text)

                    except:
                        continue
                
                if best_result:
                    logger.debug(f"OCR result: '{best_result}'")
                    
                    # Validate MM:SS format
                    if ':' in best_result:
                        parts = best_result.split(':')
                        if len(parts) == 2 and all(p.isdigit() for p in parts if p):
                            self.result = best_result

                        else:
                            self.result = None

                    elif best_result.isdigit():
                        self.result = best_result

                    else:
                        self.result = None

                else:
                    logger.debug("OCR: No valid result")
                    self.result = None
                
            except Exception as e:
                logger.debug(f"OCR error: {e}")
                self.result = None
    
    def process(self, roi):
        """Submit ROI for processing (non-blocking)"""
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()  # Drop old request

            except:
                pass
        self.input_queue.put(roi)
    
    def get_result(self):
        """Get the latest result (non-blocking)"""
        return self.result
    
    def stop(self):
        self.running = False

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
        
        # Start OCR worker thread
        self.ocr_worker = OCRWorker()
        self.ocr_worker.start()
        self.last_energy_value = None
        
        logger.info("OCR worker thread started")

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

    def _extract_energy_roi(self, frame, energy_loc, energy_size):
        """Extract energy ROI and submit to OCR worker (non-blocking)"""
        eh, ew = energy_size
        ex, ey = energy_loc
        
        # The energy number appears immediately to the right of the energy icon
        # Adjust positioning to capture just the timer text
        gap_start = 5  # Small gap between icon and text
        gap_width = 80  # Width to capture "00:47" format
        
        roi_x = ex + ew + gap_start
        roi_y = ey
        roi_w = gap_width
        roi_h = eh
        
        h, w = frame.shape[:2]
        if roi_x + roi_w > w or roi_y + roi_h > h:
            return
            
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
        
        # Draw debug rectangle on display frame to visualize ROI
        # This helps verify we're capturing the right area
        
        # Submit to OCR worker (non-blocking)
        self.ocr_worker.process(roi)

    def _detect_all_go_buttons(self, frame):
        """Detect ALL green GO buttons using color detection - returns list of detections"""
        h, w = frame.shape[:2]
        
        search_top = int(h * 0.10)
        search_bottom = int(h * 0.30)
        search_roi = frame[search_top:search_bottom, 0:w]
        
        hsv = cv2.cvtColor(search_roi, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([45, 120, 120])
        upper_green = np.array([75, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        go_buttons = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 1500 < area < 8000:
                x, y, w_btn, h_btn = cv2.boundingRect(cnt)
                
                aspect_ratio = w_btn / h_btn if h_btn > 0 else 0
                if 1.8 < aspect_ratio < 3.5:
                    go_buttons.append({
                        'x': x,
                        'y': y + search_top,
                        'w': w_btn,
                        'h': h_btn,
                        'area': area
                    })
        
        go_buttons.sort(key=lambda btn: btn['x'])
        
        return go_buttons

    def process_frame(self, frame):
        if frame is None: return
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        
        board_clogged = self._is_board_full(frame)

        # 1. ENERGY (GREEN) - Non-blocking OCR
        energy_roi_drawn = False
        if self.tm.energy_template is not None:
            res_e = cv2.matchTemplate(frame, self.tm.energy_template, cv2.TM_CCOEFF_NORMED)
            _, max_val_e, _, max_loc_e = cv2.minMaxLoc(res_e)
            if max_val_e >= self.energy_thresh:
                eh, ew = self.tm.energy_template.shape[:2]
                cv2.rectangle(display_frame, max_loc_e, (max_loc_e[0]+ew, max_loc_e[1]+eh), (0, 255, 0), self.mark_thickness)
                
                # Draw ROI rectangle for debugging (cyan color)
                gap_start = 5
                gap_width = 60
                roi_x = max_loc_e[0] + ew + gap_start
                roi_y = max_loc_e[1]
                # cv2.rectangle(display_frame, (roi_x, roi_y), 
                #             (roi_x + gap_width, roi_y + eh), (255, 255, 0), 2)
                energy_roi_drawn = True
                
                # Submit energy ROI to OCR worker (non-blocking)
                self._extract_energy_roi(frame, max_loc_e, (eh, ew))
        
        # Get latest OCR result (non-blocking)
        ocr_result = self.ocr_worker.get_result()
        if ocr_result is not None:
            self.last_energy_value = ocr_result

        # 2. ALL GO BUTTONS - COLOR-BASED DETECTION
        go_buttons = self._detect_all_go_buttons(frame)
        
        if go_buttons:
            for idx, btn in enumerate(go_buttons):
                x, y = btn['x'], btn['y']
                w_btn, h_btn = btn['w'], btn['h']
                
                padding = 5
                x_padded = max(0, x - padding)
                y_padded = max(0, y - padding)
                w_padded = min(w, x + w_btn + padding) - x_padded
                h_padded = min(h, y + h_btn + padding) - y_padded
                
                color = (0, 0, 255)
                cv2.rectangle(display_frame, (x_padded, y_padded), 
                            (x_padded + w_padded, y_padded + h_padded), color, 2)
                
                label = f"GO{idx+1}"
                cv2.putText(display_frame, label, (x_padded, y_padded - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 3. ITEMS (Generators & Orders)
        for category in ['generators', 'orders']:
            for item in self.tm.categories[category]:
                res_i = cv2.matchTemplate(frame, item['img'], cv2.TM_CCORR_NORMED, mask=item['mask'])
                _, max_val_i, _, max_loc_i = cv2.minMaxLoc(res_i)
                if max_val_i >= self.threshold:
                    ih, iw = item['img'].shape[:2]
                    color = (0, 255, 0) if category == 'generators' else (0, 165, 255)
                    cv2.rectangle(display_frame, max_loc_i, (max_loc_i[0]+iw, max_loc_i[1]+ih), color, self.mark_thickness)

        # UI OVERLAY - Show energy and GO buttons count
        y_pos = 100
        
        # Display Energy (use cached value) - show as string to preserve format
        if self.last_energy_value is not None:
            energy_display = str(self.last_energy_value)
            cv2.putText(display_frame, f"Energy: {energy_display}", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 128), 2)
            y_pos += 40

        else:
            # Show "Detecting..." while OCR is working
            cv2.putText(display_frame, "Energy: ...", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            y_pos += 40
        
        # Display GO buttons count
        status_text = f"GO Buttons: {len(go_buttons)}" if go_buttons else "GO Buttons: 0"
        if board_clogged:
            status_text += " | Board: FULL"

        # cv2.putText(display_frame, status_text, (20, y_pos), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("Bot Tracker", cv2.resize(display_frame, None, fx=self.scale_percent, fy=self.scale_percent))
        cv2.waitKey(1)
    
    def cleanup(self):
        """Stop OCR worker thread"""
        if hasattr(self, 'ocr_worker'):
            self.ocr_worker.stop()
            logger.info("OCR worker stopped")
