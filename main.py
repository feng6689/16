import os
import sys
import cv2
from video_reader import VideoReader
from watermark_detector import WatermarkDetector
from mask_generator import MaskGenerator
from video_annotator import VideoAnnotator
from statistics import StatisticsAnalyzer

def main():
    input_video = "1.mp4"
    output_video = "2.mp4"
    mask_output_dir = "1"
    stats_output_file = "statistics.txt"
    
    if not os.path.exists(input_video):
        print(f"错误: 输入视频文件 {input_video} 不存在")
        sys.exit(1)
    
    print("="*60)
    print("视频水印检测程序启动")
    print("="*60)
    print(f"输入视频: {input_video}")
    print(f"输出视频: {output_video}")
    print(f"掩码输出目录: {mask_output_dir}")
    print("="*60)
    
    try:
        video_reader = VideoReader(input_video)
        frame_count, fps, width, height = video_reader.open_video()
        
        print(f"\n视频信息:")
        print(f"  总帧数: {frame_count}")
        print(f"  帧率: {fps:.2f} FPS")
        print(f"  分辨率: {width}x{height} 像素")
        print("\n开始处理视频...\n")
        
        watermark_detector = WatermarkDetector(width, height)
        mask_generator = MaskGenerator(mask_output_dir)
        video_annotator = VideoAnnotator(output_video, fps, width, height)
        stats_analyzer = StatisticsAnalyzer()
        
        processed_frames = 0
        detected_frames = 0
        
        for frame_index, frame in video_reader.read_frames():
            watermark_mask = watermark_detector.detect_watermark(frame, frame_index)
            
            mask_generator.save_mask(watermark_mask, frame_index)
            
            annotated_frame = video_annotator.annotate_frame(frame, watermark_mask, frame_index)
            video_annotator.write_frame(annotated_frame)
            
            if watermark_mask.sum() > 0:
                contours, _ = cv2.findContours(watermark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    stats_analyzer.add_watermark_position(frame_index, x, y, w, h)
                detected_frames += 1
            
            processed_frames += 1
            if processed_frames % 10 == 0:
                print(f"处理进度: {processed_frames}/{frame_count} 帧 ({processed_frames*100/frame_count:.1f}%)")
        
        print(f"\n处理完成!")
        print(f"总处理帧数: {processed_frames}")
        print(f"检测到水印的帧数: {detected_frames}")
        
        video_reader.release()
        video_annotator.release()
        
        stats = stats_analyzer.print_statistics()
        stats_analyzer.save_statistics_to_file(stats_output_file)
        
        print(f"掩码图像已保存到: {os.path.abspath(mask_output_dir)}")
        print(f"标注视频已保存到: {os.path.abspath(output_video)}")
        print(f"统计报告已保存到: {os.path.abspath(stats_output_file)}")
        
        print("\n" + "="*60)
        print("程序执行完成!")
        print("="*60)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
