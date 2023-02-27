import gym
import minerl
from findcave.config import LOCAL_DEDUG
import numpy as np
import cv2
if LOCAL_DEDUG:
    from minerl.human_play_interface.human_play_interface import HumanPlayInterface
else:
    import aicrowd_gym
UPDOWNANGLE_CLIP = 35
STUCK_ROTATE_ANGLE = 30


class FindCaveEnvWarper():
    def __init__(self, env_name):
        if LOCAL_DEDUG:
            self.env = gym.make(env_name)
            # self.env = HumanPlayInterface(self.env)
        else:
            self.env = aicrowd_gym.make(env_name)
        self.action_space = self.env.action_space
        self.updown_angle = 0
        self.is_stuck = False
        # avoid mamga
        self.rotate_count = 0
        self.water_lower = np.array([100, 43, 46])
        self.water_upper = np.array([124, 255, 255])
        self.magma_lower = np.array([8, 202, 100])
        self.magma_upper = np.array([16, 255, 240])
        self.rotate_to_right = False
        self.rotate_to_left = False
        self.continus_rotate_count = 0
    def reset(self):
        self.updown_angle = 0
        self.is_stuck = False
        # self.env.seed(42)
        # print(self.env._seed)
        # self.env.reset()
        # self.env.seed(42)
        # print(self.env._seed)

        return self.env.reset()

    def detect_danger_area(self, image):
        if image is None:
            self.continus_rotate_count = 0
            self.rotate_to_left = False
            self.rotate_to_right = False
            return
        # box = [290, 180, 350, 360]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # bluemask = cv2.inRange(hsv, self.water_lower, self.water_upper)[180:, 100:540]
        magmamask = cv2.inRange(hsv, self.magma_lower, self.magma_upper)[180:, 100:540]  # [180:, 100:540]

        left_magma = magmamask[:, :220].mean()/255
        right_magma = magmamask[:, 220:].mean()/255
        if magmamask.mean() / 255.0 > 0.03:
            if left_magma >= right_magma:
                self.rotate_to_right = True
            else:
                self.rotate_to_left = True
            self.continus_rotate_count += 1
        else:
            self.continus_rotate_count = 0
            self.rotate_to_left = False
            self.rotate_to_right = False
        if LOCAL_DEDUG:
            print('avoid: ',left_magma,right_magma, self.continus_rotate_count, self.rotate_to_left, self.rotate_to_right)
        # if self.continus_rotate_count>10:
        #     self.rotate_to_left = False
        #     self.rotate_to_right = False

    def step(self, action, image=None, override_if_human_input: bool = False):

        if self.updown_angle > UPDOWNANGLE_CLIP:
            action['camera'][0] += UPDOWNANGLE_CLIP-self.updown_angle
        elif self.updown_angle < -UPDOWNANGLE_CLIP:
            action['camera'][0] += -UPDOWNANGLE_CLIP - self.updown_angle
        if LOCAL_DEDUG:
            print('camera11: ', action['camera'])
        self.updown_angle += action['camera'][0]
        self.updown_angle = max(-90, self.updown_angle)
        self.updown_angle = min(90, self.updown_angle)

        self.detect_danger_area(image)
        if self.is_stuck:
            # if action['camera'][1]>=0:
            #     action['camera'][1] += STUCK_ROTATE_ANGLE
            # else:
            #     action['camera'][1] -= STUCK_ROTATE_ANGLE
            if LOCAL_DEDUG:
                print("mystuckmystuckmystuckmystuckmystuckmystuck")
            action['camera'][1] += STUCK_ROTATE_ANGLE
            self.is_stuck = False
        if self.rotate_to_left:
            action['camera'][1] = -10 # STUCK_ROTATE_ANGLE
        elif self.rotate_to_right:
            action['camera'][1] = 10

        if LOCAL_DEDUG:
            print('camera22: ', action['camera'])
        return self.env.step(action)
    def render(self):
        self.env.render()
    def close(self):
        self.env.close()