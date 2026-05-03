import cv2
import numpy as np
from collections import deque

class WatermarkDetector:
    def __init__(self, frame_width, frame_height, history_frames=50):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.history_frames = history_frames
        
        self.frame_history = deque(maxlen=history_frames)
        self.gray_history = deque(maxlen=history_frames)
        
        self.watermark_positions = []
        self.frame_count = 0
        
        self.cumulative_diff = np.zeros((frame_height, frame_width), dtype=np.float32)
        
        self.min_watermark_area = 50
        self.max_watermark_area = frame_width * frame_height * 0.15
        self.min_aspect_ratio = 0.15
        self.max_aspect_ratio = 15.0
        
        self.detected_regions_history = deque(maxlen=20)
        
    def _compute_multi_frame_diff(self):
        if len(self.gray_history) < 3:
            return None
        
        cumulative = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        
        for i in range(1, len(self.gray_history)):
            diff = cv2.absdiff(self.gray_history[i], self.gray_history[i-1])
            cumulative += diff.astype(np.float32)
        
        cumulative = cumulative / len(self.gray_history)
        
        return cumulative.astype(np.uint8)
    
    def _detect_moving_regions(self):
        if len(self.gray_history) < 5:
            return None
        
        motion_masks = []
        
        for i in range(1, len(self.gray_history)):
            diff = cv2.absdiff(self.gray_history[i], self.gray_history[i-1])
            _, binary = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
            motion_masks.append(binary)
        
        if not motion_masks:
            return None
        
        persistent_motion = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        for mask in motion_masks:
            persistent_motion += mask.astype(np.float32)
        
        threshold = len(motion_masks) * 0.3
        _, persistent_mask = cv2.threshold(persistent_motion, threshold, 255, cv2.THRESH_BINARY)
        
        return persistent_mask.astype(np.uint8)
    
    def _detect_text_regions(self, gray):
        mser = cv2.MSER_create(_delta=5, _min_area=50, _max_area=5000)
        regions, _ = mser.detectRegions(gray)
        
        text_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.drawContours(text_mask, [hull], 0, 255, -1)
        
        return text_mask
    
    def _detect_high_contrast_regions(self, gray):
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        gradient_magnitude = gradient_magnitude.astype(np.uint8)
        
        _, binary_gradient = cv2.threshold(gradient_magnitude, 30, 255, cv2.THRESH_BINARY)
        
        return binary_gradient
    
    def _detect_semi_transparent_overlay(self):
        if len(self.gray_history) < 10:
            return None
        
        gray_frames = np.array(list(self.gray_history), dtype=np.float32)
        
        variance = np.var(gray_frames, axis=0)
        
        variance = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX)
        variance = variance.astype(np.uint8)
        
        _, binary_variance = cv2.threshold(variance, 10, 255, cv2.THRESH_BINARY)
        
        return binary_variance
    
    def _track_region_movement(self, current_regions):
        if len(self.detected_regions_history) < 2:
            self.detected_regions_history.append(current_regions)
            return current_regions
        
        previous_regions = self.detected_regions_history[-1]
        
        consistent_regions = []
        
        for curr in current_regions:
            cx, cy, cw, ch = curr
            curr_center = (cx + cw/2, cy + ch/2)
            
            for prev in previous_regions:
                px, py, pw, ph = prev
                prev_center = (px + pw/2, py + ph/2)
                
                distance = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                   (curr_center[1] - prev_center[1])**2)
                
                size_similarity = abs(cw - pw) / max(cw, pw) < 0.3 and abs(ch - ph) / max(ch, ph) < 0.3
                
                if distance < 200 and size_similarity:
                    consistent_regions.append(curr)
                    break
        
        self.detected_regions_history.append(current_regions)
        
        return consistent_regions
    
    def _morphological_operations(self, mask):
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)
        kernel_horizontal = np.ones((1, 15), np.uint8)
        kernel_vertical = np.ones((15, 1), np.uint8)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        dilated_h = cv2.dilate(mask, kernel_horizontal, iterations=1)
        dilated_v = cv2.dilate(mask, kernel_vertical, iterations=1)
        
        connected = cv2.bitwise_or(dilated_h, dilated_v)
        
        return connected
    
    def _filter_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_mask = np.zeros_like(mask)
        valid_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_watermark_area < area < self.max_watermark_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio:
                    valid_regions.append((x, y, w, h))
                    cv2.rectangle(filtered_mask, (x, y), (x + w, y + h), 255, -1)
        
        return filtered_mask, valid_regions
    
    def _merge_adjacent_regions(self, regions, distance_threshold=80):
        if not regions:
            return []
        
        merged = []
        used = [False] * len(regions)
        
        for i in range(len(regions)):
            if used[i]:
                continue
            
            x1, y1, w1, h1 = regions[i]
            current_group = [(x1, y1, w1, h1)]
            used[i] = True
            
            for j in range(i + 1, len(regions)):
                if used[j]:
                    continue
                
                x2, y2, w2, h2 = regions[j]
                
                dx = min(abs(x1 + w1 - x2), abs(x2 + w2 - x1), abs(x1 - x2))
                dy = min(abs(y1 + h1 - y2), abs(y2 + h2 - y1), abs(y1 - y2))
                
                same_row = abs(y1 - y2) < max(h1, h2) * 0.5
                close_enough = dx < distance_threshold
                
                if same_row and close_enough:
                    current_group.append((x2, y2, w2, h2))
                    used[j] = True
            
            if current_group:
                x_coords = [c[0] for c in current_group]
                y_coords = [c[1] for c in current_group]
                w_coords = [c[2] for c in current_group]
                h_coords = [c[3] for c in current_group]
                
                x_min = min(x_coords)
                y_min = min(y_coords)
                x_max = max(x_coords) + max(w_coords)
                y_max = max(y_coords) + max(h_coords)
                
                merged.append((x_min, y_min, x_max - x_min, y_max - y_min))
        
        return merged
    
    def detect_watermark(self, frame, frame_index):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        self.frame_history.append(frame.copy())
        self.gray_history.append(gray.copy())
        
        if len(self.gray_history) < 5:
            return np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        
        motion_mask = self._detect_moving_regions()
        
        text_mask = self._detect_text_regions(gray)
        
        gradient_mask = self._detect_high_contrast_regions(gray)
        
        variance_mask = self._detect_semi_transparent_overlay()
        
        combined_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        
        if motion_mask is not None:
            combined_mask = cv2.bitwise_or(combined_mask, motion_mask)
        
        combined_mask = cv2.bitwise_or(combined_mask, text_mask)
        combined_mask = cv2.bitwise_or(combined_mask, gradient_mask)
        
        if variance_mask is not None:
            combined_mask = cv2.bitwise_and(combined_mask, variance_mask)
        
        processed_mask = self._morphological_operations(combined_mask)
        
        filtered_mask, valid_regions = self._filter_contours(processed_mask)
        
        consistent_regions = self._track_region_movement(valid_regions)
        
        merged_regions = self._merge_adjacent_regions(consistent_regions)
        
        if not merged_regions and valid_regions:
            merged_regions = self._merge_adjacent_regions(valid_regions)
        
        watermark_mask = np.zeros_like(filtered_mask)
        
        for x, y, w, h in merged_regions:
            cv2.rectangle(watermark_mask, (x, y), (x + w, y + h), 255, -1)
            
            self.watermark_positions.append({
                'frame_index': frame_index,
                'x': x,
                'y': y,
                'width': w,
                'height': h
            })
        
        self.frame_count += 1
        
        return watermark_mask
    
    def get_watermark_statistics(self):
        if not self.watermark_positions:
            return {
                'avg_width': 0,
                'avg_height': 0,
                'x_min': 0,
                'x_max': 0,
                'y_min': 0,
                'y_max': 0
            }
        
        widths = [pos['width'] for pos in self.watermark_positions]
        heights = [pos['height'] for pos in self.watermark_positions]
        x_coords = [pos['x'] for pos in self.watermark_positions]
        y_coords = [pos['y'] for pos in self.watermark_positions]
        
        return {
            'avg_width': np.mean(widths),
            'avg_height': np.mean(heights),
            'x_min': np.min(x_coords),
            'x_max': np.max(x_coords) + max(widths),
            'y_min': np.min(y_coords),
            'y_max': np.max(y_coords) + max(heights)
        }
