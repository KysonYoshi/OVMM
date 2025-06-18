import numpy as np
import skimage


from mapper import build_mapper
from fmm_planner import FMMPlanner

class LocalAgent:
    def __init__(
        self,
        map_size_mm,
        map_resolution,
        robot_pos,
    ):
        self.mapper = build_mapper()
        self.map_size_mm = map_size_mm
        self.map_resolution = map_resolution
        self.robot_pos = robot_pos
        self.initialize_local_map_pose()
        self.reset_goal = True


    def initialize_local_map_pose(self):
        self.mapper.reset_map()
        self.x_gt, self.y_gt, self.o_gt = (
            self.map_size_mm / 2.0,
            self.map_size_mm / 2.0,
            self.robot_pos["first_pos"]["o"], ################################
        )
        self.reset_goal = True

    def update_local_map(self, latest_height_map):
        
        self.local_map, self.local_exp_map, self.wall_map_gt = self.mapper.update_map(
            latest_height_map, (self.x_gt, self.y_gt, self.o_gt)
        )

    def set_goal(self, delta_dist, delta_rot):
        start = (
            int(self.x_gt / self.map_resolution),
            int(self.y_gt / self.map_resolution),
        )
        #print(self.x_gt, self.map_resolution, start[0], start[0])
        goal = (
            start[0]
            + int(
                delta_dist * np.cos(delta_rot + self.o_gt) * 1000.0 / self.map_resolution
            ),
            start[1]
            + int(
                delta_dist * np.sin(delta_rot + self.o_gt) * 1000.0 / self.map_resolution
            ),
        )
        print("goal: ", goal[0], goal[1])
        self.goal = goal

    def navigate_local(self):
        traversible = (
            skimage.morphology.binary_dilation(
                self.local_map, skimage.morphology.disk(5)
            )
            != True
        )
        start = (
            int(self.x_gt / self.map_resolution),
            int(self.y_gt / self.map_resolution),
        )
        try:
            traversible[start[0] - 2 : start[0] + 3, start[1] - 2 : start[1] + 3] = 1
        except:
            import ipdb

            ipdb.set_trace()
        planner = FMMPlanner(traversible, 360 // 10, 1)

        if self.reset_goal:
            planner.set_goal(self.goal, auto_improve=True)
            self.goal = planner.get_goal()
            self.reset_goal = False
        else:
            planner.set_goal(self.goal, auto_improve=True)

        stg_x, stg_y = start
        #print(stg_x, stg_y )
        stg_x, stg_y, replan = planner.get_short_term_goal2((stg_x, stg_y))
        #print(stg_x, stg_y )
        
        def get_l2_distance(x1, x2, y1, y2):
            """
            Computes the L2 distance between two points.
            """
            print("Distance: ", ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        
        if get_l2_distance(start[0], self.goal[0], start[1], self.goal[1]) < 3:
            terminate = 1
        else:
            terminate = 0

        agent_orientation = np.rad2deg(self.o_gt)
        action = planner.get_next_action(start, (stg_x, stg_y), agent_orientation)
        self.stg_x, self.stg_y = int(stg_x), int(stg_y)
        return action, terminate
    
    
