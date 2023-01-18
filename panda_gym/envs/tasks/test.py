from typing import Any, Dict

import datetime
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import multiprocessing
# from queue import Queue
import random
import cv2
import os

from panda_gym.envs.core import Task
from panda_gym.utils import distance
import tacto
import hydra
import time
import pybulletX as px
import pybullet as p 
from omegaconf import DictConfig, OmegaConf
from sys import platform

#conf_path = "/home/tanguy/Documents/Project_rob/panda-gym/test/conf/grasp.yaml"
#conf_path = "/home/julien/roboticProject/panda-gym/test/conf/grasp.yaml"
#Path config for windows julien 
conf_path = "C:/Users/bouff/RoboticsProject/panda-gym/test/conf/grasp.yaml"
class Test(Task):
    def __init__(
        self,
        sim,
        robot,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,
    ) -> None:
        super().__init__(sim)

        #self.all_processes = []
        self.reward_type = reward_type
        self.conf_path = OmegaConf.load(conf_path)
        self.distance_threshold = distance_threshold
        self.robot = robot
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        self.path_obj = ""
        self.df = pd.DataFrame(columns=['rgbdigits','depthdigits','rgbcam','depthcam','touching','timestamp'])
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

        #Create digit feedback
        self.digits = tacto.Sensor(**self.conf_path.tacto)
        p.resetDebugVisualizerCamera(**self.conf_path.pybullet_camera)
        id = 1
        links_number = [11, 14]
        self.digits.add_camera(id, links_number)

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
            half_extents=[0.2,0.35,0.02],
            mass=0,
            position= [0,0,0.1],
            rgba_color = [0,0,0],
            ghost = False,
        )
        
        # self.sim.create_sphere(
        #     body_name="target",
        #     radius=0.02,
        #     mass=0.0,
        #     ghost=True,
        #     position=np.zeros(3),
        #     rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        # )
        
        #insert object here

        self.sim.loadURDF(
            body_name="target",
            fileName=self.path_obj,
            basePosition=np.array([-0.3,0.5,0.3]),
            baseOrientation=np.array([0,0,0,1]),
            useFixedBase=False,
        )

    def get_obs(self) -> np.ndarray: 

        # compute the RGB image and depth
        rgbdigit, depthdigit = self.return_digit_data()
        rgbcam, depthcam = self.return_camera()
        contact = self.detectcollision()

        #obs_vec = [rgbdigit,depthdigit,rgbcam,depthcam,contact]

        self.df = self.df.append({'rgbdigits':rgbdigit,
                                  'depthdigits':depthdigit,
                                  'rgbcam':rgbcam,
                                  'depthcam':depthcam,
                                  'touching':contact,
                                  'timestamp':datetime.datetime.now()},
                                  ignore_index=True)

        return np.array([])  # no tasak-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.robot.get_ee_position())
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
        rgb_tiny = np.reshape(images[2], (self.height_camera, self.width_camera, 4)) #* 1. / 255.
        # Convert the RGB image to BGR
        rgb_tiny = cv2.cvtColor(rgb_tiny, cv2.COLOR_BGR2RGB)

        
        # for process in self.all_processes:
        #     process.terminate()

        cv2.imshow("RGB image", rgb_tiny)
        cv2.imshow("Depth image", depth_tiny)
        

        return rgb_tiny, depth_tiny
        
    def return_digit_data(self):
        color, depth = self.digits.render()
        self.digits.updateGUI(color, depth)
        return np.array(color), np.array(depth)

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

    def detectcollision(self):
        contact_points = p.getContactPoints(self.sim._bodies_idx['Doosan'],self.sim._bodies_idx['target'])

        if len(contact_points) > 0:
            collision_detected = 1
        else:
            collision_detected = 0

        return np.array([collision_detected])

