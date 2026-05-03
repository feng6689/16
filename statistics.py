import numpy as np
from datetime import datetime

class StatisticsAnalyzer:
    def __init__(self):
        self.watermark_positions = []
        
    def add_watermark_position(self, frame_index, x, y, width, height):
        self.watermark_positions.append({
            'frame_index': frame_index,
            'x': x,
            'y': y,
            'width': width,
            'height': height
        })
    
    def calculate_statistics(self):
        if not self.watermark_positions:
            return {
                'total_frames': 0,
                'avg_width': 0,
                'avg_height': 0,
                'avg_area': 0,
                'x_min': 0,
                'x_max': 0,
                'y_min': 0,
                'y_max': 0,
                'x_range': 0,
                'y_range': 0,
                'total_movement': 0
            }
        
        widths = [pos['width'] for pos in self.watermark_positions]
        heights = [pos['height'] for pos in self.watermark_positions]
        x_coords = [pos['x'] for pos in self.watermark_positions]
        y_coords = [pos['y'] for pos in self.watermark_positions]
        
        avg_width = np.mean(widths)
        avg_height = np.mean(heights)
        avg_area = avg_width * avg_height
        
        x_min = np.min(x_coords)
        x_max = np.max([pos['x'] + pos['width'] for pos in self.watermark_positions])
        y_min = np.min(y_coords)
        y_max = np.max([pos['y'] + pos['height'] for pos in self.watermark_positions])
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        total_movement = 0
        for i in range(1, len(self.watermark_positions)):
            prev_x = self.watermark_positions[i-1]['x']
            prev_y = self.watermark_positions[i-1]['y']
            curr_x = self.watermark_positions[i]['x']
            curr_y = self.watermark_positions[i]['y']
            
            distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            total_movement += distance
        
        return {
            'total_frames': len(self.watermark_positions),
            'avg_width': avg_width,
            'avg_height': avg_height,
            'avg_area': avg_area,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'x_range': x_range,
            'y_range': y_range,
            'total_movement': total_movement
        }
    
    def print_statistics(self):
        stats = self.calculate_statistics()
        
        print("\n" + "="*60)
        print("水印检测统计结果")
        print("="*60)
        print(f"检测到水印的帧数: {stats['total_frames']}")
        print(f"水印区域平均尺寸: {stats['avg_width']:.2f} x {stats['avg_height']:.2f} 像素")
        print(f"水印区域平均面积: {stats['avg_area']:.2f} 像素²")
        print("\n移动轨迹范围:")
        print(f"  横向范围: 从 x = {stats['x_min']:.2f} 到 x = {stats['x_max']:.2f}")
        print(f"  纵向范围: 从 y = {stats['y_min']:.2f} 到 y = {stats['y_max']:.2f}")
        print(f"  横向移动距离: {stats['x_range']:.2f} 像素")
        print(f"  纵向移动距离: {stats['y_range']:.2f} 像素")
        print(f"  总移动距离: {stats['total_movement']:.2f} 像素")
        print("="*60 + "\n")
        
        return stats
    
    def save_statistics_to_file(self, filepath):
        stats = self.calculate_statistics()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"统计生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n")
            f.write("水印检测统计结果\n")
            f.write("="*60 + "\n")
            f.write(f"检测到水印的帧数: {stats['total_frames']}\n")
            f.write(f"水印区域平均尺寸: {stats['avg_width']:.2f} x {stats['avg_height']:.2f} 像素\n")
            f.write(f"水印区域平均面积: {stats['avg_area']:.2f} 像素²\n")
            f.write("\n移动轨迹范围:\n")
            f.write(f"  横向范围: 从 x = {stats['x_min']:.2f} 到 x = {stats['x_max']:.2f}\n")
            f.write(f"  纵向范围: 从 y = {stats['y_min']:.2f} 到 y = {stats['y_max']:.2f}\n")
            f.write(f"  横向移动距离: {stats['x_range']:.2f} 像素\n")
            f.write(f"  纵向移动距离: {stats['y_range']:.2f} 像素\n")
            f.write(f"  总移动距离: {stats['total_movement']:.2f} 像素\n")
            f.write("="*60 + "\n")
        
        print(f"统计结果已保存到: {filepath}")
