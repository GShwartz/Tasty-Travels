import cv2

class TaskBot:
    def __init__(self, adb_ctrl, template_mgr):
        self.adb = adb_ctrl
        self.tm = template_mgr
        self.threshold = 0.88
        self.mark_thickness = 2
        # ADJUST THIS: 0.5 makes the window half the original height
        self.scale_percent = 0.6 

    def process_frame(self, frame):
        display_frame = frame.copy()
        found_gen_ids = []

        # --- ENERGY TRACKING ---
        if self.tm.energy_template is not None:
            res = cv2.matchTemplate(frame, self.tm.energy_template, cv2.TM_CCORR_NORMED, mask=self.tm.energy_mask)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= 0.95:
                h, w = self.tm.energy_template.shape[:2]
                cv2.rectangle(display_frame, max_loc, (max_loc[0]+w, max_loc[1]+h), (0, 255, 0), self.mark_thickness)

        # --- GENERATORS TRACKING ---
        for item in self.tm.categories['generators']:
            res = cv2.matchTemplate(frame, item['img'], cv2.TM_CCORR_NORMED, mask=item['mask'])
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= self.threshold:
                found_gen_ids.append(item['id'])
                h, w = item['img'].shape[:2]
                cv2.rectangle(display_frame, max_loc, (max_loc[0]+w, max_loc[1]+h), (0, 255, 0), self.mark_thickness)

        # --- ORDERS TRACKING ---
        for item in self.tm.categories['orders']:
            res = cv2.matchTemplate(frame, item['img'], cv2.TM_CCORR_NORMED, mask=item['mask'])
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= self.threshold:
                h, w = item['img'].shape[:2]
                is_met = item['id'] in found_gen_ids
                color = (255, 0, 0) if is_met else (0, 165, 255)
                cv2.rectangle(display_frame, max_loc, (max_loc[0]+w, max_loc[1]+h), color, self.mark_thickness)

        # --- RESIZE LOGIC ---
        # Calculate new dimensions to make it shorter
        width = int(display_frame.shape[1] * self.scale_percent)
        height = int(display_frame.shape[0] * self.scale_percent)
        dim = (width, height)
        
        resized_frame = cv2.resize(display_frame, dim, interpolation=cv2.INTER_AREA)
                
        cv2.imshow("Bot Tracker", resized_frame)
        cv2.waitKey(1)
