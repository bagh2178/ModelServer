import pygame
import numpy as np
import time
import open3d as o3d
import threading
import queue
from .robot_interface import GO2Interface
from multiprocessing import Process


class GUIController:
    def __init__(self, robot:GO2Interface):
        self.robot = robot

        self.robot_stage = 'running'
        self.vx, self.vy, self.vz = 0, 0, 0
        self.vyaw = 0
        self.running = True

        self.vx_goal, self.vy_goal = 0, 0
        self.coordinate_state = 0

        self.pointcloud_queue = queue.Queue(maxsize=10)
        self.rgb_queue = queue.Queue(maxsize=10)
        self.pointcloud_thread = None
        self.rgb_thread = None
        self.thread_lock = threading.Lock()

        self.pc  = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='GO2 PointCloud',
                               width=256, height=144)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        opt.point_size = 2.0 
        opt.show_coordinate_frame = True
        opt.light_on = True 
        self.update_fre = 0
        
        # initialize the pygame
        pygame.init()
        self.screen = pygame.display.set_mode((256, 144))
        pygame.display.set_caption("宇树Go2机器狗远程控制器")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        
        # key_value
        self.key_states = {
            pygame.K_w: 0,  # forward
            pygame.K_s: 0,  # backward
            pygame.K_a: 0,  # moveleft
            pygame.K_d: 0,  # moveright
            pygame.K_q: 0,  # turn left
            pygame.K_e: 0,  # turn right
            pygame.K_j: 0,  # move according to the point position
        }
    
    def start_threads(self):
        """
        open the render thread
        """
        self.pointcloud_thread = threading.Thread(target=self._pointcloud_render_loop, daemon=True)
        self.rgb_thread = threading.Thread(target=self._rgb_render_loop, daemon=True)
        
        self.pointcloud_thread.start()
        self.rgb_thread.start()
    
    def stop_threads(self):
        """
        stop all the render thread
        """
        self.running = False
        if self.pointcloud_thread and self.pointcloud_thread.is_alive():
            self.pointcloud_thread.join(timeout=1.0)
        if self.rgb_thread and self.rgb_thread.is_alive():
            self.rgb_thread.join(timeout=1.0)
    
    def _pointcloud_render_loop(self):
        """
        point cloud render thread
        """
        try:
            while self.running:
                try:
                    xyz = self.pointcloud_queue.get(timeout=1.0)
                    if xyz is not None and len(xyz):
                        
                        if len(xyz) > 0:
                            with self.thread_lock:
                                self.pc.points = o3d.utility.Vector3dVector(xyz)
                                
                                colors = self._generate_pointcloud_colors(xyz)
                                self.pc.colors = o3d.utility.Vector3dVector(colors)

                                if not hasattr(self, "_view_inited"):
                                    self.vis.add_geometry(self.pc)
                                    self._init_view_control()
                                    self._view_inited = True

                                self.vis.update_geometry(self.pc)

                            self.vis.poll_events()
                            self.vis.update_renderer()
                            
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"point cloud render thread error: {e}")
                    continue
        finally:
            print("point cloud render thread end")
    
    def _init_view_control(self):
        ctr = self.vis.get_view_control()
        ctr.set_front([0, 1, 0])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, -1])
        ctr.set_zoom(0.3)
    
    def _generate_pointcloud_colors(self, xyz):
        if len(xyz) == 0:
            return np.array([[0.5, 0.5, 0.5]])
        
        center = np.mean(xyz, axis=0)
        distances = np.linalg.norm(xyz - center, axis=1)
        max_dist = np.max(distances)

        if max_dist > 0:
            normalized_distances = distances / max_dist
            
            colors = np.zeros((len(xyz), 3))
            colors[:, 0] = normalized_distances  # farther, redder
            colors[:, 1] = 0.2 
            colors[:, 2] = 1.0 - normalized_distances 
            
            colors = np.clip(colors, 0, 1)
        else:
            colors = np.array([[0.2, 0.7, 1.0]] * len(xyz))
        
        return colors
    
    def _rgb_render_loop(self):
        """
        RGB image render thread
        """
        try:
            while self.running:
                try:
                    image = self.rgb_queue.get(timeout=1.0)
                    if image is not None:
                        with self.thread_lock:
                            surf = pygame.surfarray.make_surface(image)
                            surf = pygame.transform.scale(surf, (256, 144))
                            self.screen.blit(surf, (0, 0))
                            pygame.display.flip()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"RGB render thread error: {e}")
                    continue
        finally:
            print("RGB render thread end")
    
    def rollout(self):
        """
        control loop
        """
        try:
            # render thread
            self.start_threads()
            
            while self.running:
                self._handle_events()
                # self.move_x_y_yaw()
                self._update_data_queues()
                self.clock.tick(30)  # FPS
        finally:
            self.stop_threads()
            self.robot.stop()
            self.vis.destroy_window()
            pygame.quit()
    
    def _update_data_queues(self):
        """
        update the date queue for render thread
        """
        # point cloud data
        if self.update_fre % 5 == 0:
            self.update_fre = 0
            xyz = self.robot.get_pointcloud()
            if xyz is not None:
                xyz = xyz[np.isfinite(xyz).all(1)]
                if len(xyz) > 0:
                    try:
                        self.pointcloud_queue.put_nowait(xyz)
                    except queue.Full:
                        try:
                            self.pointcloud_queue.get_nowait()
                            self.pointcloud_queue.put_nowait(xyz)
                        except queue.Empty:
                            pass
        
        # update RGB image data
        image = self.robot.get_camera_image()
        if image is not None:
            try:
                self.rgb_queue.put_nowait(image)
            except queue.Full:
                try:
                    self.rgb_queue.get_nowait()
                    self.rgb_queue.put_nowait(image)
                except queue.Empty:
                    pass
        
        self.update_fre += 1

    def _handle_events(self):
        '''
        handle the pygame events
        '''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                break
            
            # key down
            elif event.type == pygame.KEYDOWN:
                if event.key in self.key_states:
                    self.key_states[event.key] = 1
                elif event.key == pygame.K_SPACE:  # switch the robot stage
                    if self.robot_stage == 'running':
                        self.robot_stage = 'stop'
                        self.robot.stop()
                    elif self.robot_stage == 'stop':
                        self.robot_stage = 'running'
                        self.robot.start(1)
                elif event.key == pygame.K_ESCAPE:  # exit
                    self.running = False
            
            # key up
            elif event.type == pygame.KEYUP:
                if event.key in self.key_states:
                    self.key_states[event.key] = 0
            
            # get the relative motion of the mouse
            dx, dy = pygame.mouse.get_rel()

    # def _render(self):
    #     # point cloud
    #     if self.update_fre % 5 == 0:
    #         self.update_fre = 0
    #         xyz = self.robot.get_pointcloud()
    #         xyz = xyz[np.isfinite(xyz).all(1)]
    #         if xyz is not None and len(xyz):
    #             self.pc.points = o3d.utility.Vector3dVector(xyz)
    #             self.pc.colors = o3d.utility.Vector3dVector(np.array([[0.2, 0.7, 1.0]] * len(xyz)))
    #             # self.vis.remove_geometry(self.pc)
    #             if not hasattr(self, "_view_inited"):
    #                 self.vis.add_geometry(self.pc)
    #                 self._view_inited = True
    #             self.vis.update_geometry(self.pc)
    #         self.vis.poll_events()
    #         self.vis.update_renderer()
    #     self.update_fre += 1

    #     # rgb
    #     image = self.robot.get_camera_image()
    #     if image is not None:
    #         surf  = pygame.surfarray.make_surface(image)
    #         surf  = pygame.transform.scale(surf, (256, 144))
    #         self.screen.blit(surf, (0, 0))
        
    #     pygame.display.flip()

    def move_x_y_yaw(self):
        '''
        move the robot in x, y, yaw direction
        '''
        if self.robot.robot_stage is not None:
            print(self.robot.robot_stage.position, self.robot.robot_stage.imu_state.rpy)
        if (self.key_states[pygame.K_j] == 1 or self.coordinate_state == 1) and self.robot.robot_stage is not None:
            self.coordinate_state = self.robot.get_to_coordination_goal(0, 0.5)

        if self.coordinate_state == 0:
            # vx
            if self.key_states[pygame.K_w] == 0 and self.key_states[pygame.K_s] == 0:
                self.vx = 0.0
            else:
                self.vx += np.clip(0.05 * (self.key_states[pygame.K_w] - self.key_states[pygame.K_s]), -0.5, 0.5)
            # vy
            if self.key_states[pygame.K_a] == 0 and self.key_states[pygame.K_d] == 0:
                self.vy = 0.0
            else:
                self.vy += np.clip(0.05 * (self.key_states[pygame.K_a] - self.key_states[pygame.K_d]), -0.5, 0.5)
            # vyaw
            if self.key_states[pygame.K_q] == 0 and self.key_states[pygame.K_e] == 0:
                self.vyaw = 0.0
            else:
                self.vyaw += np.clip(0.075 * (self.key_states[pygame.K_q] - self.key_states[pygame.K_e]), -0.5, 0.5)
            
            self.robot.move(self.vx, self.vy, self.vyaw)

        
