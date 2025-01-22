import pyrealsense2 as rs
import numpy as np
import cv2
import pyorbbecsdk as ob
from scipy.ndimage import zoom


class OrbbecCamera():
    def __init__(self, serial_number):
        self.serial_number = serial_number
        ctx = ob.Context()
        device_list = ctx.query_devices()
        self.device = device_list.get_device_by_serial_number(self.serial_number)
        self.pipeline = ob.Pipeline(self.device)
        self.device = self.pipeline.get_device()
        self.device_info = self.device.get_device_info()
        self.device_name = self.device_info.get_name()
        config = ob.Config()

        depth_profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        color_profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        if self.device_name == 'Femto Bolt':
            self.device.set_bool_property(ob.OBPropertyID(2000), True)  # OB_PROP_COLOR_AUTO_EXPOSURE_BOOL
            self.device.set_int_property(ob.OBPropertyID(2001), 200)  # OB_PROP_COLOR_EXPOSURE_INT
            self.device.set_int_property(ob.OBPropertyID(2002), 0)  # OBPropertyID.OB_PROP_COLOR_GAIN_INT
            self.device.set_bool_property(ob.OBPropertyID(2003), False)  # OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL
            self.device.set_int_property(ob.OBPropertyID(2004), 4000)  # OB_PROP_COLOR_WHITE_BALANCE_INT
            # depth_profile = depth_profile_list.get_video_stream_profile(512, 512, ob.OBFormat.Y16, 30)
            depth_profile = depth_profile_list.get_video_stream_profile(640, 576, ob.OBFormat.Y16, 30)
            color_profile = color_profile_list.get_video_stream_profile(1280, 720, ob.OBFormat.RGB, 30)
            # color_profile = profile_list.get_video_stream_profile(1920, 1080, ob.OBFormat.RGB, 30)
            # color_profile = profile_list.get_video_stream_profile(3840, 2160, ob.OBFormat.RGB, 30)
            self.depth_scale = 0.001
        elif self.device_name == 'Orbbec Gemini 336L':
            self.device.set_bool_property(ob.OBPropertyID(2000), True)  # OB_PROP_COLOR_AUTO_EXPOSURE_BOOL
            self.device.set_int_property(ob.OBPropertyID(2001), 200)  # OB_PROP_COLOR_EXPOSURE_INT
            self.device.set_int_property(ob.OBPropertyID(2002), 0)  # OBPropertyID.OB_PROP_COLOR_GAIN_INT
            self.device.set_bool_property(ob.OBPropertyID(2003), False)  # OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL
            self.device.set_int_property(ob.OBPropertyID(2004), 4000)  # OB_PROP_COLOR_WHITE_BALANCE_INT
            depth_profile = depth_profile_list.get_video_stream_profile(1280, 720, ob.OBFormat.Y16, 30)
            color_profile = color_profile_list.get_video_stream_profile(1280, 720, ob.OBFormat.RGB, 30)
            self.depth_scale = 0.001
        config.enable_stream(depth_profile)
        config.enable_stream(color_profile)
        config.set_align_mode(ob.OBAlignMode.SW_MODE)

        self.pipeline.start(config)
        self.pipeline.enable_frame_sync()

        for _ in range(50):
            self.pipeline.wait_for_frames(100)

    def capture_rgbd_image(self, zoom_factor=1.0):
        try:
            frames = self.pipeline.wait_for_frames(100)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            timestamp = 0.5 * (color_frame.get_system_timestamp_us() + depth_frame.get_system_timestamp_us())
            timestamp = 1e-6 * timestamp

            width = color_frame.get_width()
            height = color_frame.get_height()
            color_image = np.asanyarray(color_frame.get_data())
            color_image = np.resize(color_image, (height, width, 3))

            width = depth_frame.get_width()
            height = depth_frame.get_height()
            depth_image = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_image = depth_image.reshape((height, width))

            # color_image = zoom(color_image, (zoom_factor, zoom_factor, 1))
            # depth_image = zoom(depth_image, (zoom_factor, zoom_factor, 1))

            return color_image, depth_image, timestamp
        except:
            self.pipeline.stop()

    def capture_point_cloud(self):
        try:
            frames = self.pipeline.wait_for_frames(100)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            timestamp = 0.5 * (color_frame.get_system_timestamp_us() + depth_frame.get_system_timestamp_us())
            timestamp = 1e-6 * timestamp
            camera_param = self.pipeline.get_camera_param()
            points = frames.get_point_cloud(camera_param)
            points[:, :3] = points[:, :3] * self.depth_scale
            filtered_points = points[~np.all(points[:, :3] == 0, axis=1)]
            return filtered_points, timestamp
        except:
            self.pipeline.stop()

    def capture_color_point_cloud(self):
        try:
            frames = self.pipeline.wait_for_frames(100)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            timestamp = 0.5 * (color_frame.get_system_timestamp_us() + depth_frame.get_system_timestamp_us())
            timestamp = 1e-6 * timestamp
            camera_param = self.pipeline.get_camera_param()
            points = frames.get_color_point_cloud(camera_param)
            points[:, :3] = points[:, :3] * self.depth_scale
            filtered_points = points[~np.all(points[:, :3] == 0, axis=1)]
            return filtered_points, timestamp
        except:
            self.pipeline.stop()

    def capture_rgbd_and_color_point_cloud(self, zoom_factor=1.0):
        try:
            frames = self.pipeline.wait_for_frames(100)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            timestamp = 0.5 * (color_frame.get_system_timestamp_us() + depth_frame.get_system_timestamp_us())
            timestamp = 1e-6 * timestamp

            width = color_frame.get_width()
            height = color_frame.get_height()
            color_image = np.asanyarray(color_frame.get_data())
            color_image = np.resize(color_image, (height, width, 3))

            width = depth_frame.get_width()
            height = depth_frame.get_height()
            depth_image = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_image = depth_image.reshape((height, width))

            camera_param = self.pipeline.get_camera_param()
            points = frames.get_color_point_cloud(camera_param)
            points[:, :3] = points[:, :3] * self.depth_scale
            filtered_points = points[~np.all(points[:, :3] == 0, axis=1)]

            # color_image = zoom(color_image, (zoom_factor, zoom_factor, 1))
            # depth_image = zoom(depth_image, (zoom_factor, zoom_factor, 1))

            return color_image, depth_image, filtered_points, timestamp
        except:
            self.pipeline.stop()

    def capture_camera_param(self):
        camera_param = self.pipeline.get_camera_param()
        return camera_param

    def capture_intrinsic(self):
        fx = self.pipeline.get_camera_param().rgb_intrinsic.fx
        fy = self.pipeline.get_camera_param().rgb_intrinsic.fy
        cx = self.pipeline.get_camera_param().rgb_intrinsic.cx
        cy = self.pipeline.get_camera_param().rgb_intrinsic.cy
        color_intrinsics = (fx, fy, cx, cy)
        
        fx = self.pipeline.get_camera_param().depth_intrinsic.fx
        fy = self.pipeline.get_camera_param().depth_intrinsic.fy
        cx = self.pipeline.get_camera_param().depth_intrinsic.cx
        cy = self.pipeline.get_camera_param().depth_intrinsic.cy
        depth_intrinsics = (fx, fy, cx, cy)

        return color_intrinsics, depth_intrinsics




if __name__ == '__main__':
    serial_number = 'CL8M841006W'  # 替换为你实际的设备序列号
    camera = OrbbecCamera(serial_number)
    while True:
        # color_image, depth_image = camera.capture_rgbd_image()
        color_image, depth_image = camera.capture_color_point_cloud()
        # color_intrinsics, depth_intrinsics = camera.capture_intrinsic('327122078142')
        # print('color_intrinsics', color_intrinsics)
        # print('depth_intrinsics', depth_intrinsics)

        # 显示图像
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('RGB Image', color_image)
        # depth_image_display = (depth_image * 255 / np.max(depth_image)).astype(np.uint8)  # 归一化并转换回uint8以显示
        # cv2.imshow('Depth Image', depth_image_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()