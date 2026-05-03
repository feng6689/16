import cv2
import numpy as np
import os

class VideoReader:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        
    def open_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")
        
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return self.frame_count, self.fps, self.width, self.height
    
    def read_frames(self):
        if not self.cap:
            raise ValueError("视频未打开，请先调用open_video()方法")
        
        frame_index = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            yield frame_index, frame
            frame_index += 1
    
    def release(self):
        if self.cap:
            self.cap.release()
    
    def __del__(self):
        self.release()
