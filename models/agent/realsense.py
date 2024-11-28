import pyrealsense2 as rs
import numpy as np
import cv2

def get_specific_camera(serial_number):
    # 创建一个上下文对象，用于管理RealSense设备
    ctx = rs.context()
    devices = ctx.query_devices()

    for dev in devices:
        if dev.get_info(rs.camera_info.serial_number) == serial_number:
            return dev
    raise ValueError(f"未找到指定序列号({serial_number})的设备")

def capture_rgbd_image(serial_number):
    # 获取特定序列号的设备
    device = get_specific_camera(serial_number)
    
    # 配置管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    
    # 开始流
    profile = pipeline.start(config)
    
    try:
        # 等待一帧到达
        for i in range(50):
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            raise RuntimeError("无法获取有效的深度或颜色帧")

        # 将图像转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 将深度图从uint16转为float32，并归一化
        depth_image = depth_image.astype(np.float32) / 65536.0  # 归一化到 [0, 1]
        depth_image = np.expand_dims(depth_image, axis=-1)  # 添加通道维度

        return color_image, depth_image

    finally:
        # 停止流
        pipeline.stop()


if __name__ == '__main__':
    # 使用示例
    serial_number = '337322070914'  # 替换为你实际的设备序列号
    rgb_image, depth_image = capture_rgbd_image(serial_number)

    # 显示图像
    cv2.imshow('RGB Image', rgb_image)
    depth_image_display = (depth_image * 255).astype(np.uint8)  # 转换回uint8以显示
    cv2.imshow('Depth Image', depth_image_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
