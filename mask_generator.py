import cv2
import os
import numpy as np

class MaskGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self._ensure_output_dir()
        
    def _ensure_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建输出目录: {self.output_dir}")
    
    def save_mask(self, mask, frame_index):
        filename = f"{frame_index:06d}.png"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, mask)
        return filepath
    
    def save_mask_with_info(self, mask, frame_index, watermark_info):
        filepath = self.save_mask(mask, frame_index)
        
        info_filename = f"{frame_index:06d}_info.txt"
        info_filepath = os.path.join(self.output_dir, info_filename)
        
        with open(info_filepath, 'w', encoding='utf-8') as f:
            f.write(f"帧序号: {frame_index}\n")
            f.write(f"水印区域: x={watermark_info['x']}, y={watermark_info['y']}, "
                   f"width={watermark_info['width']}, height={watermark_info['height']}\n")
            f.write(f"掩码尺寸: {mask.shape[1]}x{mask.shape[0]}\n")
        
        return filepath, info_filepath
    
    def get_mask_count(self):
        if not os.path.exists(self.output_dir):
            return 0
        
        mask_files = [f for f in os.listdir(self.output_dir) 
                     if f.endswith('.png') and f.split('.')[0].isdigit()]
        return len(mask_files)
