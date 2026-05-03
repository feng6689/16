import cv2
import numpy as np

class WatermarkDetector:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.prev_gray = None
        self.watermark_positions = []
        
    def detect_watermark(self, frame, frame_index):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        
        frame_diff = cv2.absdiff(self.prev_gray, gray)
        
        _, binary_diff = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)
        
        edges = cv2.Canny(gray, 50, 150)
        
        _, binary_edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_abs = np.absolute(laplacian)
        _, binary_hf = cv2.threshold(laplacian_abs, 10, 255, cv2.THRESH_BINARY)
        binary_hf = binary_hf.astype(np.uint8)
        
        combined_mask = cv2.bitwise_or(binary_diff, binary_edges)
        combined_mask = cv2.bitwise_or(combined_mask, binary_hf)
        
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        watermark_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        watermark_bboxes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < self.frame_width * self.frame_height * 0.1:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 5.0:
                    watermark_bboxes.append((x, y, w, h))
                    cv2.rectangle(watermark_mask, (x, y), (x + w, y + h), 255, -1)
        
        if watermark_bboxes:
            x_coords = [bbox[0] for bbox in watermark_bboxes]
            y_coords = [bbox[1] for bbox in watermark_bboxes]
            w_coords = [bbox[2] for bbox in watermark_bboxes]
            h_coords = [bbox[3] for bbox in watermark_bboxes]
            
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords) + max(w_coords)
            y_max = max(y_coords) + max(h_coords)
            
            self.watermark_positions.append({
                'frame_index': frame_index,
                'x': x_min,
                'y': y_min,
                'width': x_max - x_min,
                'height': y_max - y_min
            })
        
        self.prev_gray = gray.copy()
        
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
