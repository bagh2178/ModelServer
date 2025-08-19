from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import (
    PointCloud2_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
    PathPoint_,
    SportModeState_,
    LowState_,
    LowCmd_,
)
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
)
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_
)
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.utils.thread import RecurrentThread
from .param import *
from collections import deque
from multiprocessing import Process, Queue, Manager
from queue import Empty, Full
import numpy as np
import cv2
import time


def get_robot_information(robot_state, point_cloud, low_state, 
                          ip_address, camera_queue: Queue, command_queue: Queue):
    '''
    use the interface of unitree to subscribe the robot information topic
    '''
    def _handle_robot_state(msg: SportModeState_):
        try:
            if robot_state.full():
                try:
                    robot_state.get_nowait()
                except:
                    pass
            robot_state.put(msg, block=False)
        except:
            pass     

    def _handle_point_cloud(msg: PointCloud2_):
        '''
        Handle the point cloud message
        '''
        names, fmts, offs = [], [], []
        endian = '>' if msg.is_bigendian else '<'
        for f in msg.fields:
            names.append(f.name)
            fmts.append(endian + DT_MAP[f.datatype])
            offs.append(f.offset)

        dtype = np.dtype({'names'   : names,
                          'formats' : fmts,
                          'offsets' : offs,
                          'itemsize': msg.point_step})
        data_bytes = bytes(msg.data)
        arr = np.frombuffer(data_bytes, dtype=dtype)
        xyz = np.vstack((arr['x'], arr['y'], arr['z'])).T.astype(np.float32)
        pc_buffer.append(xyz)
        pc = np.vstack(pc_buffer)

        try:
            if point_cloud.full():
                try:
                    point_cloud.get_nowait()
                except:
                    pass
            point_cloud.put(pc, block=False)
        except:
            pass

    def _handle_low_state(msg: LowState_):
        try:
            if low_state.full():
                try:
                    low_state.get_nowait()
                except:
                    pass
            low_state.put(msg, block=False)
        except:
            pass

    def _low_cmd_write():
        if low_cmd:
            lowcmd_publisher.Write(low_cmd)

    # initialize the DDS
    ChannelFactoryInitialize(0, ip_address)

    # initialize the video client
    camera_client = VideoClient()
    camera_client.SetTimeout(5.0)
    camera_client.Init()
    print('camera client initialized')

    # initialize the high level client
    sport_client = SportClient()
    sport_client.SetTimeout(5.0) # timeout
    sport_client.Init()
    print('sport client initialized')

    # create publisher
    lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
    lowcmd_publisher.Init()

    pc_buffer = deque(maxlen=10)

    sport_state_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sport_state_subscriber.Init(_handle_robot_state, 0)

    point_cloud_subscriber = ChannelSubscriber("rt/utlidar/cloud", PointCloud2_)
    point_cloud_subscriber.Init(_handle_point_cloud)

    lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    lowstate_subscriber.Init(_handle_low_state, 10)
    low_cmd = None

    print('start the subscribe process')   

    try:
        while True:
            try:
                command = command_queue.get_nowait()
                if command[0] == 'Move':
                    sport_client.Move(command[1], command[2], command[3])
                elif command[0] == 'BalanceStand':
                    sport_client.BalanceStand()
                elif command[0] == 'RecoveryStand':
                    sport_client.RecoveryStand()
                elif command[0] == 'StopMove':
                    sport_client.StopMove()
                elif command[0] == 'StandDown':
                    sport_client.StandDown()
                elif command[0] == 'GetImageSample':
                    code, data = camera_client.GetImageSample()
                    if camera_queue.full():
                        try:
                            camera_queue.get_nowait()
                        except:
                            pass
                    camera_queue.put((code, data), block=False)
                elif command[0] == 'TrajectoryFollow':
                    sport_client.TrajectoryFollow(command[1])
                elif command[0] == 'LowCommand_Write':
                    low_cmd = command[1]
                elif command[0] == '_low_state_init':
                    msc = MotionSwitcherClient()
                    msc.SetTimeout(5.0)
                    msc.Init()

                    # make sure the robot is in the low_control mode
                    _, result = msc.CheckMode()
                    while result['name']:
                        sport_client.StandDown()
                        msc.ReleaseMode()
                        _, result = msc.CheckMode()
                        time.sleep(1)

                    lowCmdWriteThreadPtr = RecurrentThread(
                        interval=0.002, target=_low_cmd_write, name="directjointcontrol"
                    )
                    lowCmdWriteThreadPtr.Start()
            except:
                pass

            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    

class GO2Interface:
    def __init__(self, ip_address):
        '''
        Initialize the connection to the robot
        Args:
            ip_address: the network card name of the robot
        '''
        self.ip_address = ip_address
        self.camera_queue = Queue(maxsize=1)
        self.command_queue = Queue(maxsize=100)

        self.robot_state = Queue(maxsize=1)
        self.point_cloud = Queue(maxsize=1)
        self.low_state   = Queue(maxsize=1)

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.InitLowCmd()
        self.crc = CRC()

        self.start()

    def InitLowCmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # PMSM模式
            self.low_cmd.motor_cmd[i].q = PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def subscribe_process(self):
        p = Process(
            target=get_robot_information,
            args=(
                self.robot_state, self.point_cloud, self.low_state, 
                self.ip_address, self.camera_queue, self.command_queue,
            )
        )
        p.start()  

    def move(self, vx, vy, vyaw):
        '''
        vx:   [-2.5~3.8] (m/s)
        vy:   [-1.0~1.0] (m/s)
        vyaw: [-4~4] (rad/s)
        '''
        self.command_queue.put(('Move', vx, vy, vyaw))

    def start(self, _index=0):
        # enter the state of balance
        if _index == 0:
            self.command_queue.put(('BalanceStand',))
            print('robot has started')
            # print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
            # input("Press Enter to continue...")
        elif _index == 1:
            print('robot has started')
            self.command_queue.put(('RecoveryStand',))

    def stop(self):
        self.command_queue.put(('StopMove',))
        self.command_queue.put(('BalanceStand',))
        self.command_queue.put(('StandDown',))
        print('robot has stopped')

    def get_camera_image(self):
        self.command_queue.put(('GetImageSample',))
        try:
            (code, data) = self.camera_queue.get(timeout=1.0)
            if code == 0 and data is not None:
                image_data = np.frombuffer(bytes(data), dtype=np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                if image is not None:
                    image = image[:,:,[2,1,0]]
                    image = cv2.transpose(image)
                return image
        except Empty:
            pass
        except Exception as e:
            print(f"camera wrong: {e}")
        else:
            return None

    def get_to_coordination_goal(self, x, y):
        '''
        using pid to control the robot to the coordination goal
        yaw: [-3.14, 3.14]
        '''
        robot_state = self.get_sportstate()
        if robot_state == None:
            return 0
        x0, y0 = robot_state.position[0], robot_state.position[1]
        dx, dy = x - x0, y - y0
        yaw = np.arctan2(dy, dx)
        yaw0 = robot_state.imu_state.rpy[2]
        dyaw = yaw - yaw0

        dis = np.sqrt(dx**2 + dy**2)
        dx = dis * np.cos(dyaw)
        dy = dis * np.sin(dyaw)
        if dis < 0.05:
            self.move(0, 0, 0)
            return 0
        else:
            vyaw = np.clip(dyaw, -0.75, 0.75)
            vx = np.clip(0.1 * dx, -0.5, 0.5) + np.sign(dx) * 0.2
            vy = np.clip(0.1 * dy, -0.5, 0.5) + np.sign(dy) * 0.2
            
            self.move(vx, vy, vyaw)
            return 1

    def follow_coordination(self, x, y, yaw=624):
        '''
        get to the coordination
        '''
        if np.abs(yaw) > 2 * np.pi:
            robot_state = self.get_sportstate()
            if robot_state is not None:
                pos_list = np.array(robot_state.position)
                yaw = float(pos_list[2]) if len(pos_list) >= 3 else 0.0
            else:
                yaw = 0.0

        point = PathPoint_
        point.x = x
        point.y = y
        point.yaw = yaw
        point.t_from_start = np.sqrt(x**2 + y**2)
        point.vx = 0
        point.vy = 0
        point.vyaw = 0

        self.follow_trajactory([point])

    def follow_trajactory(self, path: list):
        '''
        Follow the trajectory, the path must be a list of PathPoint_
        '''
        print('follow trajactory')
        self.command_queue.put(('TrajectoryFollow', path))

    def joint_control(self, joint_pos: list):
        '''
        Control the joint position, create the channel firstly
        '''
        if not hasattr(self, "_low_state_init"):
            self.command_queue.put(('_low_state_init',))
            self._low_state_init = True
            time.sleep(0.1)
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = joint_pos[i]
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = KP
            self.low_cmd.motor_cmd[i].kd = KD
            self.low_cmd.motor_cmd[i].tau = 0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.command_queue.put(('LowCommand_Write', self.low_cmd))

    def get_lowstate(self):
        try:
            low_state = self.low_state.get(timeout=1.0)
            return low_state
        except Exception as e:
            return None

    def get_pointcloud(self):
        try:
            point_cloud = self.point_cloud.get(timeout=1.0)
            return point_cloud
        except Exception as e:
            return None

    def get_sportstate(self):
        try:
            robot_state = self.robot_state.get(timeout=1.0)
            return robot_state
        except Exception as e:
            return None
