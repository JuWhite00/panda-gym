from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from queue import Queue
import random
import cv2
import os

from panda_gym.envs.core import Task
from panda_gym.envs.robots import doosan
from panda_gym.utils import distance
import tacto
import hydra
import time
import pybulletX as px
import pybullet as p 
from omegaconf import DictConfig, OmegaConf
from sys import platform

conf_path = "/home/tanguy/Documents/Project_rob/panda-gym/test/conf/grasp.yaml"
#conf_path = "/home/julien/roboticProject/panda-gym/test/conf/grasp.yaml"
#Path config for windows julien 
# conf_path = "C:/Users/bouff/RoboticsProject/panda-gym/test/conf/grasp.yaml"
class Test(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,
    ) -> None:
        super().__init__(sim)

        #self.all_processes = []
        self.reward_type = reward_type
        self.conf_path = OmegaConf.load(conf_path)
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        self.path_obj = ""
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

        # create the camera view
        self.width_camera = 128
        self.height_camera = 128

        self.fov = 60
        self.aspect = self.width_camera / self.height_camera
        self.near = 0.02
        self.far = 1

        self.view_matrix_camera = p.computeViewMatrix([0, 0, 0.5], [0, 0, 0], [1, 0, 0])
        self.projection_matrix_camera = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)

    def _create_scene(self) -> None:
        
        dirname = os.path.join(os.path.split(os.path.split(os.path.split(os.path.split(__file__)[0])[0])[0])[0])

        if self.path_obj == "":
            if platform == "win32":
                self.path_obj = dirname + '\\mesh\\pybullet-URDF-models\\urdf_models\\models\\'
                select_name = random.sample(os.listdir(self.path_obj),1)[0]
                self.path_obj = self.path_obj + select_name + '\\model.urdf'
            else:
                self.path_obj = dirname + '/mesh/pybullet-URDF-models/urdf_models/models/'
                select_name = random.sample(os.listdir(self.path_obj),1)[0]
                self.path_obj = self.path_obj + select_name + '/model.urdf'
        
        #Part for julien
        # if self.path_obj == "":
        #     self.path_obj = dirname + '/mesh/pybullet-URDF-models/urdf_models/models/'
        #     select_name = random.sample(os.listdir(self.path_obj),1)[0]
        #     self.path_obj = self.path_obj + select_name + '/model.urdf'
        
        self.sim.create_plane(z_offset=-0.4)
        
        self.sim.create_box(
            body_name="table",
            half_extents=[0.3,0.1,0.05],
            mass=0,
            position= [0,1,0.5],
            rgba_color = [0,0,0],
            ghost = False,
        )
        
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        
        #insert object here

        self.sim.loadURDF(
            body_name="obj",
            fileName=self.path_obj,
            basePosition=np.zeros(3),
            baseOrientation=np.array([0,0,0,1]),
            useFixedBase=False,
        )

    def get_obs(self) -> np.ndarray: 
        
        self.return_digit_data()

        # compute the RGB image and depth
        rgb, depth = self.return_camera()

        return np.array([])  # no tasak-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.bool8)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -float(d > self.distance_threshold)
        else:
            return -d

    def return_camera(self): # compute the RGB image from the init camera + the depth converted
        images = p.getCameraImage(self.width_camera,
                          self.height_camera,
                          self.view_matrix_camera,
                          self.projection_matrix_camera,
                          shadow=True,
                          renderer=p.ER_TINY_RENDERER)
        
        depth_buffer_tiny = np.reshape(images[3], [self.width_camera, self.height_camera])
        depth_tiny = self.far * self.near / (self.far - (self.far - self.near) * depth_buffer_tiny)
        rgb_tiny = np.reshape(images[2], (self.height_camera, self.width_camera, 4)) * 1. / 255.
        rgb_tiny = rgb_tiny[:,:,:3]
        
        # for process in self.all_processes:
        #     process.terminate()

        cv2.imshow("RGB image", rgb_tiny)
        cv2.imshow("Depth image", depth_tiny)
        

        return rgb_tiny, depth_tiny
        
    def return_digit_data(self):
        digits = tacto.Sensor(**self.conf_path.tacto)
        p.resetDebugVisualizerCamera(**self.conf_path.pybullet_camera)
        id = 1
        links_number = [11, 14]
        digits.add_camera(id, links_number)
        #digits.add_object(obj)

        t = px.utils.SimulationThread(real_time_factor=1.0)
        t.start()

        
        color, depth = digits.render()
        digits.updateGUI(color, depth)
        time.sleep(0.01)

    def display_video(self, color_image, depth_image): # display camera images in get_obs()
        cv2.namedWindow("RGB")
        cv2.namedWindow("Depth")

        color_image = np.array(color_image)
        cv2.imshow("RGB", color_image)

        depth_image = np.array(depth_image)
        cv2.imshow("Depth", depth_image)

        time.sleep(1)

        cv2.destroyWindow("RGB")
        cv2.destroyWindow("Depth")

    def detectcollision(id_object):
        contact_points = p.getContactPoints(doosan.Doosan.getIDRobot(),id_object)

        if len(contact_points) > 0:
            collision_detected = 1
        else:
            collision_detected = 0

        return collision_detected

