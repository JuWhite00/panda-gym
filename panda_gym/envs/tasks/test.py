from typing import Any, Dict

import numpy as np
# import random
# import os

from panda_gym.envs.core import Task
from panda_gym.utils import distance
# from sys import platform





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
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        
        # dirname = os.path.dirname(__file__)
        # dirname = dirname[:-11]

        # if self.path_obj != "":
            
        #     if platform == "win32":
        #         self.path_obj = dirname + '\\mesh\\pybullet-URDF-models\\urdf_models\\models\\'
        #         select_name = random.sample(os.listdir(self.path_obj),1)[0]
        #         self.path_obj = self.path_obj + select_name + r'\model.urdf'
        #     else:
        #         self.path_obj = dirname + '/mesh/pybullet-URDF-models/urdf_models/models/'
        #         select_name = random.sample(os.listdir(self.path_obj),1)[0]
        #         self.path_obj = self.path_obj + select_name + '/model.urdf'
        
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
        # self.sim.loadURDF(
        #     body_name="obj",
        #     fileName=self.path_obj,
        #     basePosition=np.zeros(3),
        #     baseOrientation=np.zeros(3),
        #     useFixedBase=False,
        # )

    def get_obs(self) -> np.ndarray:
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
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)
