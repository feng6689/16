import cv2
import numpy as np

class VideoAnnotator:
    def __init__(self, output_path, fps, width, height):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not self.writer.isOpened():
            raise ValueError(f"无法创建输出视频文件: {output_path}")
    
    def annotate_frame(self, frame, watermark_mask, frame_index):
        annotated_frame = frame.copy()
        
        contours, _ = cv2.findContours(watermark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        cv2.putText(annotated_frame, f"Frame: {frame_index}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame
    
    def write_frame(self, frame):
        self.writer.write(frame)
    
    def release(self):
        if self.writer:
            self.writer.release()
    
    def __del__(self):
        self.release()
