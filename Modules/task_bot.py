import cv2
import time
import logging
import pytesseract
import numpy as np
import os
import threading
from queue import Queue, Empty
from pathlib import Path
import easyocr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_ROOT / "proc"
PROC_DIR.mkdir(parents=True, exist_ok=True)

TESSERACT_PATH = r'G:\Apps\Tesseract\tesseract.exe'
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

logger = logging.getLogger("TastyTravelsBot")


class OCRWorker(threading.Thread):
    """Background OCR worker using EasyOCR with optional pytesseract fallback."""
    def __init__(self, use_gpu=True, scale_factor=3):
        super().__init__(daemon=True)
        self.input_queue = Queue(maxsize=1)
        self.result = None
        self.lock = threading.Lock()
        self.running = True
        self.scale_factor = scale_factor
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)

    def run(self):
        while self.running:
            try:
                roi = self.input_queue.get(timeout=0.05)
                if roi is None:
                    continue

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                scaled = cv2.resize(binary, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_LINEAR)

                ocr_results = self.reader.readtext(scaled, allowlist='0123456789:')
                valid_results = [r for r in ocr_results if isinstance(r, (tuple, list)) and len(r) == 3 and r[2] > 0.35]

                text = None
                if valid_results:
                    text = max(valid_results, key=lambda r: r[2])[1]
                    text = ''.join(c for c in text if c.isdigit() or c == ':')

                if not text:
                    config = "--psm 7 -c tessedit_char_whitelist=0123456789:"
                    text = pytesseract.image_to_string(scaled, config=config).strip()
                    text = ''.join(c for c in text if c.isdigit() or c == ':')

                with self.lock:
                    self.result = text if text else None

            except Empty:
                continue
            except Exception:
                with self.lock:
                    self.result = None

    def process(self, roi):
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except Empty:
                pass
        self.input_queue.put(roi)

    def get_result(self):
        with self.lock:
            return self.result

    def stop(self):
        self.running = False


class TaskBot:
    def __init__(self, adb_ctrl, template_mgr):
        self.adb = adb_ctrl
        self.tm = template_mgr

        self.threshold = 0.85
        self.energy_thresh = 0.70

        self.mark_thickness = 3
        self.scale_percent = 0.6

        self.grid_top_y_pct = 0.38
        self.grid_bottom_y_pct = 0.88

        self.ocr_worker = OCRWorker(use_gpu=True, scale_factor=3)
        self.ocr_worker.start()
        self.last_energy_value = None

        logger.info("OCR worker thread started")

    def _is_board_full(self, frame):
        h, w = frame.shape[:2]
        y_start = int(h * self.grid_top_y_pct)
        y_end = int(h * self.grid_bottom_y_pct)
        roi = frame[y_start:y_end, 0:w]
        if roi.size == 0: 
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        density = np.count_nonzero(edges) / (roi.shape[0] * roi.shape[1])
        return density > 0.18

    def _extract_energy_roi(self, frame, energy_loc, energy_size):
        eh, ew = energy_size
        ex, ey = energy_loc
        roi_x = max(0, ex + ew - 2)
        roi_y = max(0, ey - 2)
        roi_w = ew + 70
        roi_h = eh + 4
        h, w = frame.shape[:2]
        if roi_x + roi_w > w or roi_y + roi_h > h:
            return
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
        self.ocr_worker.process(roi)

    def _detect_all_go_buttons(self, frame):
        """Detect green GO buttons using template matching + OCR verification."""
        h, w = frame.shape[:2]
        
        # GO buttons appear in top region
        search_top = int(h * 0.08)
        search_bottom = int(h * 0.20)
        search_roi = frame[search_top:search_bottom, 0:w]
        
        if search_roi.size == 0:
            return []
        
        go_buttons = []
        
        # Step 1: Template matching (if template exists)
        template_matches = []
        if self.tm.go_template is not None:
            res = cv2.matchTemplate(search_roi, self.tm.go_template, cv2.TM_CCOEFF_NORMED, mask=self.tm.go_mask)
            gh, gw = self.tm.go_template.shape[:2]
            
            # Find all matches above threshold
            loc = np.where(res >= 0.80)
            for pt in zip(*loc[::-1]):
                template_matches.append({
                    'x': pt[0],
                    'y': pt[1],
                    'w': gw,
                    'h': gh,
                    'conf': res[pt[1], pt[0]]
                })
            
            # Remove overlapping detections
            if template_matches:
                template_matches.sort(key=lambda m: m['conf'], reverse=True)
                filtered = []
                for match in template_matches:
                    overlap = False
                    for existing in filtered:
                        x_overlap = min(match['x'] + match['w'], existing['x'] + existing['w']) - max(match['x'], existing['x'])
                        y_overlap = min(match['y'] + match['h'], existing['y'] + existing['h']) - max(match['y'], existing['y'])
                        if x_overlap > match['w'] * 0.5 and y_overlap > match['h'] * 0.5:
                            overlap = True
                            break
                    if not overlap:
                        filtered.append(match)
                template_matches = filtered[:5]  # Max 5 buttons
        
        # Step 2: Color detection for backup/additional buttons
        hsv = cv2.cvtColor(search_roi, cv2.COLOR_BGR2HSV)
        lower_green = np.array([45, 140, 140])
        upper_green = np.array([70, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color_matches = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 1500 < area < 6000:
                x, y, w_btn, h_btn = cv2.boundingRect(cnt)
                aspect_ratio = w_btn / h_btn if h_btn > 0 else 0
                if 1.8 < aspect_ratio < 3.5:
                    color_matches.append({
                        'x': x,
                        'y': y,
                        'w': w_btn,
                        'h': h_btn
                    })
        
        # Combine template and color matches, remove duplicates
        all_candidates = template_matches + color_matches
        unique_candidates = []
        
        for candidate in all_candidates:
            is_duplicate = False
            for existing in unique_candidates:
                x_overlap = min(candidate['x'] + candidate['w'], existing['x'] + existing['w']) - max(candidate['x'], existing['x'])
                y_overlap = min(candidate['y'] + candidate['h'], existing['y'] + existing['h']) - max(candidate['y'], existing['y'])
                if x_overlap > 0 and y_overlap > 0:
                    overlap_area = x_overlap * y_overlap
                    if overlap_area > (candidate['w'] * candidate['h'] * 0.3):
                        is_duplicate = True
                        break
            if not is_duplicate:
                unique_candidates.append(candidate)
        
        # Step 3: OCR verification on each candidate
        for candidate in unique_candidates:
            x, y, w_btn, h_btn = candidate['x'], candidate['y'], candidate['w'], candidate['h']
            
            # Extract button ROI
            btn_roi = search_roi[y:y+h_btn, x:x+w_btn]
            
            if btn_roi.size == 0:
                continue
            
            # Preprocess for OCR - make text white on black
            gray = cv2.cvtColor(btn_roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            
            # Scale up for better OCR
            scale = 3
            scaled = cv2.resize(thresh, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # Run OCR in memory (no temp files)
            config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            try:
                text = pytesseract.image_to_string(scaled, config=config).strip().upper()
                
                # Check if "GO" is in the detected text
                if "GO" in text or text == "GO":
                    go_buttons.append({
                        'x': x,
                        'y': y + search_top,
                        'w': w_btn,
                        'h': h_btn,
                        'area': w_btn * h_btn
                    })
                    logger.debug(f"GO button confirmed at ({x}, {y}) with text: '{text}'")
            except Exception as e:
                logger.debug(f"OCR failed for candidate at ({x}, {y}): {e}")
                continue
        
        go_buttons.sort(key=lambda btn: btn['x'])
        
        if go_buttons:
            logger.debug(f"Total GO buttons detected: {len(go_buttons)}")
        
        return go_buttons

    def process_frame(self, frame):
        if frame is None: 
            return
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        board_clogged = self._is_board_full(frame)

        if self.tm.energy_template is not None:
            res_e = cv2.matchTemplate(frame, self.tm.energy_template, cv2.TM_CCOEFF_NORMED)
            _, max_val_e, _, max_loc_e = cv2.minMaxLoc(res_e)
            if max_val_e >= self.energy_thresh:
                eh, ew = self.tm.energy_template.shape[:2]
                cv2.rectangle(display_frame, max_loc_e, (max_loc_e[0]+ew, max_loc_e[1]+eh), (0, 255, 0), self.mark_thickness)
                self._extract_energy_roi(frame, max_loc_e, (eh, ew))

        ocr_result = self.ocr_worker.get_result()
        if ocr_result is not None:
            self.last_energy_value = ocr_result

        go_buttons = self._detect_all_go_buttons(frame)
        for idx, btn in enumerate(go_buttons):
            x, y = btn['x'], btn['y']
            w_btn, h_btn = btn['w'], btn['h']
            cv2.rectangle(display_frame, (x, y), (x+w_btn, y+h_btn), (0, 0, 255), 2)
            cv2.putText(display_frame, f"GO{idx+1}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        for category in ['generators', 'orders']:
            for item in self.tm.categories[category]:
                res_i = cv2.matchTemplate(frame, item['img'], cv2.TM_CCORR_NORMED, mask=item['mask'])
                _, max_val_i, _, max_loc_i = cv2.minMaxLoc(res_i)
                if max_val_i >= self.threshold:
                    ih, iw = item['img'].shape[:2]
                    color = (0, 255, 0) if category == 'generators' else (0, 165, 255)
                    cv2.rectangle(display_frame, max_loc_i, (max_loc_i[0]+iw, max_loc_i[1]+ih), color, self.mark_thickness)

        y_pos = 100
        cv2.putText(display_frame, f"Energy: {self.last_energy_value or '...'}", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 128), 2)

        cv2.imshow("Bot Tracker", cv2.resize(display_frame, None, fx=self.scale_percent, fy=self.scale_percent))
        cv2.waitKey(1)

    def cleanup(self):
        if hasattr(self, 'ocr_worker'):
            self.ocr_worker.stop()
            self.ocr_worker.join(timeout=1.0)
            logger.info("OCR worker stopped")
