#!/usr/bin/env python3
"""
US Camera Interface
相机读取图像的接口代码
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple, Any
import threading
import multiprocessing
from multiprocessing import Process, Queue, Value
import io
from PIL import Image

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_image(rgb_image, format='JPEG'):
    rgb_byte_io = io.BytesIO()
    if format.upper() == 'PNG':
        Image.fromarray(rgb_image).save(rgb_byte_io, format='PNG', optimize=True)
    elif format.upper() == 'JPEG':
        Image.fromarray(rgb_image).save(rgb_byte_io, format='JPEG', quality=90)
    return rgb_byte_io


def _camera_process_worker(device_id: int, camera_width: int, camera_height: int, fps: int,
                          crop_x_offset: int, crop_y_offset: int, crop_width: int, crop_height: int,
                          output_width: int, output_height: int,
                          frame_queue: Queue, running_flag: Value):
    """
    相机子进程工作函数 - 持续读取帧并放入队列
    """
    # 在子进程中配置日志
    import logging
    logging.basicConfig(level=logging.INFO)
    process_logger = logging.getLogger(f"camera_process_{device_id}")
    
    cap = None
    frame_index = 0
    try:
        # 打开相机
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            process_logger.error(f"子进程无法打开相机设备 {device_id}")
            return
        
        # 设置相机参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        process_logger.info(f"相机子进程已启动 - 设备ID: {device_id}")
        
        while running_flag.value:
            ret, frame = cap.read()
            timestamp = time.time()
            frame_index += 1
            if ret:
                # 处理图像：裁剪并缩放
                cropped_frame = frame[crop_y_offset:crop_y_offset + crop_height, 
                                    crop_x_offset:crop_x_offset + crop_width]
                resized_frame = cv2.resize(cropped_frame, (output_width, output_height))
                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                encoded_image = encode_image(resized_frame)
                
                # 尝试放入新帧（非阻塞）
                try:
                    # 如果队列满了，先取出旧帧再放入新帧
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()  # 丢弃最旧的帧
                        except:
                            pass
                    data = {
                        'color': encoded_image,
                        'timestamp': timestamp,
                        'frame_index': frame_index
                    }
                    frame_queue.put(data, block=False)
                except:
                    # 如果还是失败，跳过这一帧
                    pass
            else:
                process_logger.warning("读取帧失败")
                time.sleep(0.01)  # 短暂休息避免CPU占用过高
    
    except Exception as e:
        process_logger.error(f"相机子进程发生错误: {e}")
    
    finally:
        if cap is not None:
            cap.release()
        process_logger.info("相机子进程已结束")


class USBCamera:
    """USB Camera 相机接口类 - 使用子进程持续读取帧"""
    
    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        初始化相机
        
        Args:
            device_id: 相机设备ID (通常是0, 1, 2...)
            width: 输出图像宽度
            height: 输出图像高度
            fps: 帧率
        """
        self.device_id = device_id
        self.output_width = width
        self.output_height = height
        self.fps = fps
        self.is_opened = False
        
        # USB Camera 固定参数
        self.camera_width = 1280   # 相机原始宽度
        self.camera_height = 720   # 相机原始高度
        
        # 根据目标宽高比动态计算裁剪区域
        self._calculate_crop_parameters()
        
        # 进程相关变量
        self.camera_process = None
        self.frame_queue = None
        self.running_flag = None
        
        # 内部帧缓冲区，用于存储最新的帧并支持重复读取
        self.frame_buffer = []
        self.buffer_lock = threading.Lock()
        self.max_buffer_size = 30
    
    def _calculate_crop_parameters(self):
        """
        根据目标输出尺寸计算最佳裁剪区域，保持宽高比不变
        """
        # 边界检查：确保输出尺寸有效
        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError(f"输出尺寸必须大于0: {self.output_width}x{self.output_height}")
        
        if self.camera_width <= 0 or self.camera_height <= 0:
            raise ValueError(f"相机尺寸必须大于0: {self.camera_width}x{self.camera_height}")
        
        # 计算目标宽高比
        target_ratio = self.output_width / self.output_height
        
        # 计算原始相机的宽高比
        camera_ratio = self.camera_width / self.camera_height
        
        if target_ratio > camera_ratio:
            # 目标比例更宽，需要基于相机宽度来裁剪高度
            self.crop_width = self.camera_width
            self.crop_height = int(self.camera_width / target_ratio)
            self.crop_x_offset = 0
            self.crop_y_offset = (self.camera_height - self.crop_height) // 2
        else:
            # 目标比例更高，需要基于相机高度来裁剪宽度
            self.crop_height = self.camera_height
            self.crop_width = int(self.camera_height * target_ratio)
            self.crop_y_offset = 0
            self.crop_x_offset = (self.camera_width - self.crop_width) // 2
        
        # 确保裁剪区域不超出原始图像边界且大于0
        self.crop_width = max(1, min(self.crop_width, self.camera_width))
        self.crop_height = max(1, min(self.crop_height, self.camera_height))
        self.crop_x_offset = max(0, min(self.crop_x_offset, self.camera_width - self.crop_width))
        self.crop_y_offset = max(0, min(self.crop_y_offset, self.camera_height - self.crop_height))
        
        logger.debug(f"裁剪参数计算完成: 区域={self.crop_width}x{self.crop_height}, 偏移=({self.crop_x_offset},{self.crop_y_offset}), 目标比例={target_ratio:.3f}")
    
    def _update_frame_buffer(self):
        """
        从队列中更新内部帧缓冲区，保持帧的时间顺序
        """
        if self.frame_queue is None:
            return
        
        new_frames = []
        # 从队列中获取所有新帧
        while True:
            try:
                frame_data = self.frame_queue.get_nowait()
                new_frames.append(frame_data)
            except:
                break
        
        if new_frames:
            with self.buffer_lock:
                # 将新帧添加到缓冲区
                self.frame_buffer.extend(new_frames)
                
                # 保持缓冲区大小限制，移除最旧的帧
                if len(self.frame_buffer) > self.max_buffer_size:
                    # 按时间戳排序确保正确的时间顺序
                    self.frame_buffer.sort(key=lambda x: x['timestamp'])
                    # 只保留最新的max_buffer_size帧
                    self.frame_buffer = self.frame_buffer[-self.max_buffer_size:]
    
    def open(self) -> bool:
        """
        打开相机并启动子进程持续读取帧
        
        Returns:
            bool: 成功返回True，失败返回False
        """
        try:
            # 先测试相机是否可用
            test_cap = cv2.VideoCapture(self.device_id)
            if not test_cap.isOpened():
                logger.error(f"无法打开相机设备 {self.device_id}")
                return False
            test_cap.release()
            
            # 创建进程间通信对象
            self.frame_queue = Queue(maxsize=30)  # 保存最新的30帧
            self.running_flag = Value('i', 1)  # 进程运行标志
            
            # 启动相机子进程
            self.camera_process = Process(
                target=_camera_process_worker,
                args=(
                    self.device_id, self.camera_width, self.camera_height, self.fps,
                    self.crop_x_offset, self.crop_y_offset, self.crop_width, self.crop_height,
                    self.output_width, self.output_height,
                    self.frame_queue, self.running_flag
                )
            )
            self.camera_process.start()
            
            # 等待一小段时间确保子进程启动
            time.sleep(0.5)
            
            # 检查进程是否正常运行
            if not self.camera_process.is_alive():
                logger.error("相机子进程启动失败")
                return False
            
            logger.info(f"相机已打开 - 设备ID: {self.device_id}")
            target_ratio = self.output_width / self.output_height
            logger.info(f"裁剪区域: {self.crop_width}x{self.crop_height} (比例 {target_ratio:.2f}:1), 偏移: ({self.crop_x_offset}, {self.crop_y_offset})")
            logger.info(f"输出分辨率: {self.output_width}x{self.output_height}")
            logger.info("相机子进程已启动，开始持续读取帧")
            
            self.is_opened = True
            return True
            
        except Exception as e:
            logger.error(f"打开相机时发生错误: {e}")
            self._cleanup()
            return False
    
    def read_frame(self, num_frames: int = 1) -> Optional[list]:
        """
        从内部缓冲区读取指定数量的最新帧图像
        
        Args:
            num_frames: 要获取的帧数，范围1-30，默认为1
        
        Returns:
            Optional[list]: 包含帧数据的列表，每个元素是包含'color'和'timestamp'的字典
                          失败时返回None
        """
        if not self.is_opened or self.frame_queue is None:
            logger.warning("相机未打开或子进程未启动")
            return None
        
        # 检查子进程是否还在运行
        if self.camera_process is None or not self.camera_process.is_alive():
            logger.error("相机子进程已停止")
            return None
        
        # 限制帧数范围
        num_frames = max(1, min(num_frames, 30))
        
        try:
            # 首先更新内部缓冲区，从队列中获取最新帧
            self._update_frame_buffer()
            
            # 从内部缓冲区读取帧（不会破坏顺序，支持重复读取）
            with self.buffer_lock:
                if not self.frame_buffer:
                    logger.warning("缓冲区中没有可用帧")
                    return None
                
                # 确保缓冲区按时间戳排序
                self.frame_buffer.sort(key=lambda x: x['timestamp'])
                
                # 返回请求数量的最新帧
                if len(self.frame_buffer) >= num_frames:
                    frames = self.frame_buffer[-num_frames:]
                else:
                    frames = self.frame_buffer.copy()
                
                return frames
            
        except Exception as e:
            logger.warning(f"获取帧时出错: {e}")
            return None
    
    def get_camera_info(self) -> dict:
        """
        获取相机信息
        
        Returns:
            dict: 相机参数信息
        """
        if not self.is_opened:
            return {}
        
        target_ratio = self.output_width / self.output_height
        info = {
            'device_id': self.device_id,
            'camera_resolution': f"{self.camera_width}x{self.camera_height}",
            'output_resolution': f"{self.output_width}x{self.output_height}",
            'crop_region': f"{self.crop_width}x{self.crop_height}",
            'crop_offset': f"({self.crop_x_offset}, {self.crop_y_offset})",
            'target_ratio': f"{target_ratio:.2f}:1",
            'fps': self.fps,
            'process_alive': self.camera_process.is_alive() if self.camera_process else False,
            'queue_size': self.frame_queue.qsize() if self.frame_queue else 0,
            'buffer_size': len(self.frame_buffer) if hasattr(self, 'frame_buffer') else 0,
        }
        return info
    
    def set_property(self, prop_id: int, value: float) -> bool:
        """
        设置相机属性 (注意: 新架构下属性设置功能有限)
        
        Args:
            prop_id: 属性ID (如 cv2.CAP_PROP_BRIGHTNESS)
            value: 属性值
            
        Returns:
            bool: 设置是否成功
        """
        logger.warning("新的多进程架构下，动态属性设置功能暂不支持")
        logger.info("如需修改相机参数，请在初始化时指定或重新打开相机")
        return False
    
    def _cleanup(self):
        """清理资源"""
        try:
            # 停止子进程
            if self.running_flag is not None:
                self.running_flag.value = 0
            
            # 等待子进程结束
            if self.camera_process is not None and self.camera_process.is_alive():
                self.camera_process.join(timeout=2.0)
                if self.camera_process.is_alive():
                    logger.warning("子进程未正常结束，强制终止")
                    self.camera_process.terminate()
                    self.camera_process.join(timeout=1.0)
            
            # 清理队列
            if self.frame_queue is not None:
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        break
            
            # 清理内部缓冲区
            with self.buffer_lock:
                self.frame_buffer.clear()
            
            # 重置变量
            self.camera_process = None
            self.frame_queue = None
            self.running_flag = None
            
        except Exception as e:
            logger.error(f"清理资源时发生错误: {e}")
    
    def close(self):
        """关闭相机并停止子进程"""
        if self.is_opened:
            logger.info("正在关闭相机...")
            self._cleanup()
            self.is_opened = False
            logger.info("相机已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
    
    def __del__(self):
        """析构函数，确保资源被正确释放"""
        try:
            self.close()
        except:
            pass


def init_gopro10(device_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
    camera = None
    # 检测相机
    cameras = detect_cameras()
    if not cameras:
        print("错误: 未检测到相机设备")
        return
    
    print(f"检测到相机: {cameras}")
    
    # 使用第一个相机
    camera = USBCamera(device_id=device_id, width=width, height=height, fps=fps)
    if not camera.open():
        print("错误: 无法打开相机")
        return
    
    print("相机已成功打开!")
    print("相机信息:", camera.get_camera_info())
    return camera


def detect_cameras() -> list:
    """
    检测可用的相机设备
    
    Returns:
        list: 可用的相机设备ID列表
    """
    available_cameras = []
    
    # 检测前10个可能的设备ID
    for device_id in range(10):
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            available_cameras.append(device_id)
            cap.release()
        else:
            break
    
    logger.info(f"检测到 {len(available_cameras)} 个相机设备: {available_cameras}")
    return available_cameras


def main():
    """主函数 - 演示相机使用"""
    logger.info("开始相机测试...")
    
    # 检测可用相机
    cameras = detect_cameras()
    if not cameras:
        logger.error("未检测到任何相机设备")
        return
    
    # 使用第一个检测到的相机
    camera_id = cameras[0]
    
    # 创建相机实例
    with USBCamera(device_id=camera_id, width=640, height=480, fps=30) as camera:
        if not camera.is_opened:
            logger.error("无法打开相机")
            return
        
        # 显示相机信息
        info = camera.get_camera_info()
        logger.info(f"相机信息: {info}")
        
        logger.info("按 'q' 键退出, 按 's' 键保存图片")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # 读取帧（获取1帧）
                frame_data_list = camera.read_frame(1)
                
                if frame_data_list is None or len(frame_data_list) == 0:
                    logger.warning("无法读取帧")
                    continue
                
                # 获取最新一帧的图像数据
                frame_data = frame_data_list[0]
                encoded_frame = frame_data['color']
                timestamp = frame_data['timestamp']
                frame_index = frame_data['frame_index']
                
                # 解码图像数据
                frame = decode_image(encoded_frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转换为BGR用于OpenCV显示
                
                frame_count += 1
                
                # 在图像上添加信息
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    fps = frame_count / elapsed_time
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示图像
                cv2.imshow('USB Camera Feed', frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 保存图片
                    filename = f"camera_capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"图片已保存: {filename}")
        
        except KeyboardInterrupt:
            logger.info("用户中断程序")
        
        finally:
            cv2.destroyAllWindows()
            logger.info("程序结束")


if __name__ == "__main__":
    main()
