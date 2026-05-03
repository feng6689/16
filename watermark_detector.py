import cv2
import numpy as np

class WatermarkDetector:
    def __init__(self, frame_width, frame_height, history_frames=30):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.history_frames = history_frames
        
        self.frame_history = []
        self.background_model = None
        self.watermark_positions = []
        
        self.accumulated_mask = np.zeros((frame_height, frame_width), dtype=np.float32)
        self.frame_count = 0
        
        self.min_watermark_area = 100
        self.max_watermark_area = frame_width * frame_height * 0.1
        self.min_aspect_ratio = 0.2
        self.max_aspect_ratio = 10.0
        
    def _update_background_model(self, gray):
        self.frame_history.append(gray.astype(np.float32))
        
        if len(self.frame_history) > self.history_frames:
            self.frame_history.pop(0)
        
        self.background_model = np.mean(self.frame_history, axis=0).astype(np.uint8)
        
    def _detect_contrast_change(self, gray, background):
        contrast_gray = self._local_contrast(gray)
        contrast_bg = self._local_contrast(background)
        
        contrast_diff = cv2.absdiff(contrast_gray, contrast_bg)
        
        _, binary_contrast = cv2.threshold(contrast_diff, 5, 255, cv2.THRESH_BINARY)
        
        return binary_contrast
    
    def _local_contrast(self, gray, kernel_size=15):
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        contrast = np.zeros_like(gray, dtype=np.float32)
        for y in range(gray.shape[0]):
            for x in range(gray.shape[1]):
                local_mean = blurred[y, x]
                if local_mean > 0:
                    contrast[y, x] = abs(gray[y, x] - local_mean) / local_mean * 100
                else:
                    contrast[y, x] = 0
        
        return contrast.astype(np.uint8)
    
    def _detect_semi_transparent(self, gray, background):
        diff = cv2.absdiff(gray, background)
        
        _, binary_diff = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        
        laplacian_gray = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_gray_abs = np.absolute(laplacian_gray)
        _, binary_edges_gray = cv2.threshold(laplacian_gray_abs, 8, 255, cv2.THRESH_BINARY)
        binary_edges_gray = binary_edges_gray.astype(np.uint8)
        
        laplacian_bg = cv2.Laplacian(background, cv2.CV_64F)
        laplacian_bg_abs = np.absolute(laplacian_bg)
        _, binary_edges_bg = cv2.threshold(laplacian_bg_abs, 8, 255, cv2.THRESH_BINARY)
        binary_edges_bg = binary_edges_bg.astype(np.uint8)
        
        edges_diff = cv2.absdiff(binary_edges_gray, binary_edges_bg)
        
        combined = cv2.bitwise_or(binary_diff, edges_diff)
        
        return combined
    
    def _update_accumulated_mask(self, mask):
        self.accumulated_mask = self.accumulated_mask * 0.8 + mask.astype(np.float32) * 0.2
        
        threshold = 30
        _, temporal_mask = cv2.threshold(self.accumulated_mask, threshold, 255, cv2.THRESH_BINARY)
        
        return temporal_mask.astype(np.uint8)
    
    def _morphological_operations(self, mask):
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        return mask
    
    def _filter_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_mask = np.zeros_like(mask)
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_watermark_area < area < self.max_watermark_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio:
                    valid_contours.append((x, y, w, h, contour))
                    cv2.drawContours(filtered_mask, [contour], 0, 255, -1)
        
        return filtered_mask, valid_contours
    
    def _merge_adjacent_contours(self, contours, distance_threshold=50):
        if not contours:
            return []
        
        merged = []
        used = [False] * len(contours)
        
        for i in range(len(contours)):
            if used[i]:
                continue
            
            x1, y1, w1, h1, _ = contours[i]
            current_group = [(x1, y1, w1, h1)]
            used[i] = True
            
            for j in range(i + 1, len(contours)):
                if used[j]:
                    continue
                
                x2, y2, w2, h2, _ = contours[j]
                
                dx = min(abs(x1 + w1 - x2), abs(x2 + w2 - x1), abs(x1 - x2))
                dy = min(abs(y1 + h1 - y2), abs(y2 + h2 - y1), abs(y1 - y2))
                
                if dx < distance_threshold and dy < max(h1, h2):
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
        
        self._update_background_model(gray)
        
        if self.background_model is None or len(self.frame_history) < self.history_frames // 2:
            return np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        
        contrast_mask = self._detect_contrast_change(gray, self.background_model)
        
        semi_transparent_mask = self._detect_semi_transparent(gray, self.background_model)
        
        combined_mask = cv2.bitwise_or(contrast_mask, semi_transparent_mask)
        
        temporal_mask = self._update_accumulated_mask(combined_mask)
        
        final_mask = cv2.bitwise_and(combined_mask, temporal_mask)
        
        final_mask = self._morphological_operations(final_mask)
        
        filtered_mask, valid_contours = self._filter_contours(final_mask)
        
        merged_regions = self._merge_adjacent_contours(valid_contours)
        
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
