import numpy as np


"""
TODO:主要问题
    1. 跳下台阶的时候，轨迹不匹配，目测是实际轨迹比我们估计的快
    2. 斜着卡台阶的时候，实际是在走的，但是我们估计的可能不是斜的
    3. 在特定场景下，速度是会改变的，如水里等
"""

class AgentMap():
    DT = 0.05
    WALK_SPEED = 4.317
    SPRINT_SPEED = 5.612
    SNEAK_SPEED = 3. # TODO: fix this value
    SPRINT_JUMP_SPEED = 7.127
    
    def __init__(self, ):
        self.reset()

        self.ground_x_bias = None
        self.ground_y_bias = None

        self.direction_list = []
        self.max_n_direction_list = 4

    def reset(self, ground_truth=None):
        self.pos = np.array([0., 0.])
        self.angle = 0

        self.traj_pos = [self.pos.copy()]
        self.traj_angle = [self.angle]

        if ground_truth is not None:
            self.ground_x_bias = float(ground_truth['zpos']) 
            self.ground_y_bias = float(ground_truth['xpos'])
        self.ground_truth_traj_pos = [[0., 0.]]

    def step(self, action, ground_truth=None, visual_direction=None):
        action, speed = self.action_wrapper(action)
        # 1. step angle
        self.angle = self.limit_deg(self.angle + action['camera'][1])
        
        if not self.is_noop(visual_direction):
            # 2. step move
            if action['forward']:
                redian = self.deg2radian(self.angle)
                redian_sin = np.sin(redian)
                redian_cos = np.cos(redian)
                self.pos[0] = self.pos[0] + self.DT * speed * redian_cos
                self.pos[1] = self.pos[1] + self.DT * speed * redian_sin
            if action['left']:
                temp_angle = self.limit_deg(self.angle + 90.)
                redian = self.deg2radian(temp_angle)
                redian_sin = np.sin(redian)
                redian_cos = np.cos(redian)
                self.pos[0] = self.pos[0] + self.DT * speed * redian_cos
                self.pos[1] = self.pos[1] + self.DT * speed * redian_sin
            if action['right']:
                temp_angle = self.limit_deg(self.angle - 90.)
                redian = self.deg2radian(temp_angle)
                redian_sin = np.sin(redian)
                redian_cos = np.cos(redian)
                self.pos[0] = self.pos[0] + self.DT * speed * redian_cos
                self.pos[1] = self.pos[1] + self.DT * speed * redian_sin

            self.traj_pos.append(self.pos.copy())
            self.traj_angle.append(self.angle)

        if ground_truth is not None:
            if self.ground_x_bias is None or self.ground_y_bias is None:
                self.ground_x_bias = float(ground_truth['zpos']) 
                self.ground_y_bias = float(ground_truth['xpos'])

            self.ground_truth_traj_pos.append([
                ground_truth['zpos'] - self.ground_x_bias,
                -(ground_truth['xpos'] - self.ground_y_bias)])

    def is_noop(self, direction):
        self.direction_list.append(1 if direction else 0)
        if len(self.direction_list) > self.max_n_direction_list:
            self.direction_list.pop(0)
        
        if sum(self.direction_list) == 0:
            return True
        return False

    def action_wrapper(self, action):
        speed = self.WALK_SPEED
        if action['sprint'] == 1:
            speed = self.SPRINT_SPEED
        if action['sneak'] == 1:
            speed = self.SNEAK_SPEED
        if action['jump'] == 1 and action['sprint'] == 1:
            speed = self.SPRINT_JUMP_SPEED
            
        return action, speed

    def render(self, return_gt):
        traj_pos = np.array(self.traj_pos)
        traj_pos_gt = np.array(self.ground_truth_traj_pos)

        x_min, x_max = traj_pos[:, 0].min(), traj_pos[:, 0].max()
        y_min, y_max = traj_pos[:, 1].min(), traj_pos[:, 1].max()
        x_min_gt, x_max_gt = traj_pos_gt[:, 0].min(), traj_pos_gt[:, 0].max()
        y_min_gt, y_max_gt = traj_pos_gt[:, 1].min(), traj_pos_gt[:, 1].max()
        x_min, x_max = min(x_min, x_min_gt), max(x_max, x_max_gt)
        y_min, y_max = min(y_min, y_min_gt), max(y_max, y_max_gt)

        x_scale = (x_max - x_min) / 640
        y_scale = (y_max - y_min) / 640

        for i in range(traj_pos.shape[0]):
            traj_pos[i, 0] = max(min((traj_pos[i, 0] - x_min) / (x_scale + 1e-6), 640), 0)
            traj_pos[i, 1] = max(min((traj_pos[i, 1] - y_min) / (y_scale + 1e-6), 640), 0)
        for i in range(traj_pos_gt.shape[0]):
            traj_pos_gt[i, 0] = max(min((traj_pos_gt[i, 0] - x_min) / (x_scale + 1e-6), 640), 0)
            traj_pos_gt[i, 1] = max(min((traj_pos_gt[i, 1] - y_min) / (y_scale + 1e-6), 640), 0)

        if return_gt:
            return traj_pos, traj_pos_gt
        return traj_pos

    # =====================================================
    #                    Utils function
    # =====================================================
    def limit_deg(self, deg):
        if deg > 180.:
            deg -= 360.
        if deg < -180.:
            deg += 360.
        return deg

    def deg2radian(self, deg):
        return deg * np.pi / 180.
