import pyrealsense2 as rs
import numpy as np
import cv2


class D435i():
    def __init__(self, serial_number):
        self.serial_number = serial_number
        self.device = self.get_specific_camera()
        
        self.pipeline = rs.pipeline()
        self.align_to_color = rs.align(rs.stream.color)
        config = rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        
        self.profile = self.pipeline.start(config)

        # 创建后处理过滤器
        self.decimation = rs.decimation_filter()
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(True)

        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
        self.spatial.set_option(rs.option.filter_smooth_delta, 50)
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.1)
        self.temporal.set_option(rs.option.filter_smooth_delta, 100)

        for _ in range(50):
            self.pipeline.wait_for_frames()

    def get_specific_camera(self):
        ctx = rs.context()
        devices = ctx.query_devices()

        for dev in devices:
            if dev.get_info(rs.camera_info.serial_number) == self.serial_number:
                return dev
        raise ValueError(f"未找到指定序列号({self.serial_number})的设备")

    def capture_rgbd_image(self, zoom_factor=1.0):
        try:
            for _ in range(90):
                frames = self.pipeline.wait_for_frames()
                timestamp = frames.get_timestamp()
                timestamp = 1e-3 * timestamp
                frames = self.align_to_color.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                # 应用后处理过滤器
                # depth_frame = decimation.process(depth_frame)
                depth_frame = self.depth_to_disparity.process(depth_frame)
                depth_frame = self.spatial.process(depth_frame)
                depth_frame = self.temporal.process(depth_frame)
                depth_frame = self.disparity_to_depth.process(depth_frame)
                # depth_frame = self.hole_filling.process(depth_frame)
            
            if not depth_frame or not color_frame:
                raise RuntimeError("无法获取有效的深度或颜色帧")


            # 将图像转换为numpy数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            return color_image, depth_image, timestamp

        except:
            self.pipeline.stop()

    def capture_intrinsic(self):
        color_intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        depth_intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        return color_intrinsics, depth_intrinsics


class T265():
    def __init__(self, serial_number=None):
        self.pose_calibration_scale = 1.06
        self.fps = 200
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        if serial_number:
            self.config.enable_device(serial_number)
        
        self.config.enable_stream(rs.stream.pose)

        # Start pipeline
        profile = self.pipeline.start(self.config)

    def get_fps(self):
        return self.fps

    def get_pose(self):
        try:
            frames = self.pipeline.wait_for_frames()
            pose_frame = frames.get_pose_frame()
            timestamp = frames.get_timestamp()
            timestamp = 1e-3 * timestamp

            if not pose_frame:
                return None
            
            pose_data = pose_frame.get_pose_data()
            
            # Extract position and orientation from the pose data
            position = np.array([pose_data.translation.x, 
                                 pose_data.translation.y, 
                                 pose_data.translation.z])
            orientation = np.array([pose_data.rotation.w, 
                                    pose_data.rotation.x, 
                                    pose_data.rotation.y, 
                                    pose_data.rotation.z])

            position = position * self.pose_calibration_scale

            return position, orientation, timestamp
        
        except Exception as e:
            print(f"Error getting pose: {e}")
            return None

    def stop_pipeline(self):
        self.pipeline.stop()

if __name__ == '__main__':
    # Replace with your actual device serial number if necessary
    t265_camera = T265('119622110447')

    try:
        while True:
            position, orientation = t265_camera.get_pose()
            if position is not None and orientation is not None:
                print(f"Position: {position}, Orientation (wxyz): {orientation}")
    except KeyboardInterrupt:
        pass
    finally:
        t265_camera.stop_pipeline()





# if __name__ == '__main__':
#     serial_number = '337322070914'  # 替换为你实际的设备序列号
#     camera = D435i(serial_number)
#     # rgb_image, depth_image = camera/capture_rgbd_image(serial_number)
#     color_intrinsics, depth_intrinsics = camera.capture_intrinsic('327122078142')
#     print('color_intrinsics', color_intrinsics)
#     print('depth_intrinsics', depth_intrinsics)

    # 显示图像
    # cv2.imshow('RGB Image', rgb_image)
    # depth_image_display = (depth_image * 255 / np.max(depth_image)).astype(np.uint8)  # 归一化并转换回uint8以显示
    # cv2.imshow('Depth Image', depth_image_display)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()