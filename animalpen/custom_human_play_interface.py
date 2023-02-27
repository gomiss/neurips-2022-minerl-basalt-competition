# A simple pyglet app which controls the MineRL env,
# showing human the MineRL image and passing game controls
# to MineRL
# Intended for quick data collection without hassle or
# human corrections (agent plays but human can take over).

from typing import Optional
import time
from collections import defaultdict
import json

import numpy as np

import cv2
import gym
from gym import spaces
import pyglet
import pyglet.window.key as key

from script import Handler
from map_record import AgentMap
from animal_detector import AnimalDetector
from visual_direction import OptialFlow

# Mapping from MineRL action space names to pyglet keys
MINERL_ACTION_TO_KEYBOARD = {
    "ESC":       key.ESCAPE, # Used in BASALT to end the episode
    "attack":    pyglet.window.mouse.LEFT,
    "back":      key.S,
    "drop":      key.Q,
    "forward":   key.W,
    "hotbar.1":  key._1,
    "hotbar.2":  key._2,
    "hotbar.3":  key._3,
    "hotbar.4":  key._4,
    "hotbar.5":  key._5,
    "hotbar.6":  key._6,
    "hotbar.7":  key._7,
    "hotbar.8":  key._8,
    "hotbar.9":  key._9,
    "inventory": key.E,
    "jump":      key.SPACE,
    "left":      key.A,
    "pickItem":  pyglet.window.mouse.MIDDLE,
    "right":     key.D,
    "sneak":     key.LSHIFT,
    "sprint":    key.LCTRL,
    "swapHands": key.F,
    "use":       pyglet.window.mouse.RIGHT
}

KEYBOARD_TO_MINERL_ACTION = {v: k for k, v in MINERL_ACTION_TO_KEYBOARD.items()}


# Camera actions are in degrees, while mouse movement is in pixels
# Multiply mouse speed by some arbitrary multiplier
MOUSE_MULTIPLIER = 0.1

MINERL_FPS = 20
MINERL_FRAME_TIME = 1 / MINERL_FPS

class HumanPlayInterface(gym.Wrapper):
    def __init__(self, minerl_env):
        super().__init__(minerl_env)
        self._validate_minerl_env(minerl_env)
        self.env = minerl_env
        pov_shape = self.env.observation_space["pov"].shape
        self.window = pyglet.window.Window(
            width=pov_shape[1],
            height=pov_shape[0],
            vsync=False,
            resizable=False
        )
        self.clock = pyglet.clock.get_default()
        self.pressed_keys = defaultdict(lambda: False)
        self.window.on_mouse_motion = self._on_mouse_motion
        self.window.on_mouse_drag = self._on_mouse_drag
        self.window.on_key_press = self._on_key_press
        self.window.on_key_release = self._on_key_release
        self.window.on_mouse_press = self._on_mouse_press
        self.window.on_mouse_release = self._on_mouse_release
        self.window.on_activate = self._on_window_activate
        self.window.on_deactive = self._on_window_deactivate
        self.window.dispatch_events()
        self.window.switch_to()
        self.window.flip()

        self.last_pov = None
        self.last_mouse_delta = [0, 0]

        self.window.clear()
        self._show_message("Waiting for reset.")

        # FIXME: ===============================================
        # FIXME: yolo detector
        self.animal_detector = AnimalDetector(
            './resources/best_m.pt',
            './resources/data.yaml'
        )
        
        self.obs = None
        self.location_stats = None
        self.visual_direction = None

        self.handler = Handler(minerl_env)
        #self.agent_map = AgentMap()
        self.of = OptialFlow()
        
        # FIXME: script mode
        self.reset_angle = False

        # FIXME: run script action
        self.run_script_action = False
        self.action_index = 0
        self.obses = []

        # FIXME: pursue mode
        self.pursue_mode = False

        # FIXME: wait animal mode
        self.wait_animal_mode = False
        self.wait_animal_mode_count = 0

        # FIXME: plane detect mode
        self.plane_detect_mode = False
        self.plane_return_mode = False

        # FIXME: plane fill mode
        self.plane_fill_mode = False
        self.plane_fill_mode2 = False

        # FIXME: save video
        self.save_video = False

        # FIXME: cheat animal
        self.cheat_animal = False

        # FIXME: make hole
        self.make_hole = False

    def _on_key_press(self, symbol, modifiers):
        self.pressed_keys[symbol] = True

    def _on_key_release(self, symbol, modifiers):
        self.pressed_keys[symbol] = False

    def _on_mouse_press(self, x, y, button, modifiers):
        self.pressed_keys[button] = True

    def _on_mouse_release(self, x, y, button, modifiers):
        self.pressed_keys[button] = False

    def _on_window_activate(self):
        self.window.set_mouse_visible(False)
        self.window.set_exclusive_mouse(True)

    def _on_window_deactivate(self):
        self.window.set_mouse_visible(True)
        self.window.set_exclusive_mouse(False)

    def _on_mouse_motion(self, x, y, dx, dy):
        # Inverted
        self.last_mouse_delta[0] -= dy * MOUSE_MULTIPLIER
        self.last_mouse_delta[1] += dx * MOUSE_MULTIPLIER

    def _on_mouse_drag(self, x, y, dx, dy, button, modifier):
        # Inverted
        self.last_mouse_delta[0] -= dy * MOUSE_MULTIPLIER
        self.last_mouse_delta[1] += dx * MOUSE_MULTIPLIER

    def _validate_minerl_env(self, minerl_env):
        """Make sure we have a valid MineRL environment. Raises if not."""
        # Make sure action has right items
        remaining_buttons = set(MINERL_ACTION_TO_KEYBOARD.keys())
        remaining_buttons.add("camera")
        for action_name, action_space in minerl_env.action_space.spaces.items():
            if action_name not in remaining_buttons:
                raise RuntimeError(f"Invalid MineRL action space: action {action_name} is not supported.")
            elif (not isinstance(action_space, spaces.Discrete) or action_space.n != 2) and action_name != "camera":
                raise RuntimeError(f"Invalid MineRL action space: action {action_name} had space {action_space}. Only Discrete(2) is supported.")
            remaining_buttons.remove(action_name)
        if len(remaining_buttons) > 0:
            raise RuntimeError(f"Invalid MineRL action space: did not contain actions {remaining_buttons}")

        obs_space = minerl_env.observation_space
        if not isinstance(obs_space, spaces.Dict) or "pov" not in obs_space.spaces:
            raise RuntimeError("Invalid MineRL observation space: observation space must contain POV observation.")

    def _update_image(self, arr):
        self.window.switch_to()
        # Based on scaled_image_display.py
        image = pyglet.image.ImageData(arr.shape[1], arr.shape[0], 'RGB', arr.tobytes(), pitch=arr.shape[1] * -3)
        texture = image.get_texture()
        texture.blit(0, 0)
        self.window.flip()

    def _get_human_action(self):
        """Read keyboard and mouse state for a new action"""
        # Keyboard actions
        action = {
            name: int(self.pressed_keys[key] if key is not None else None) for name, key in MINERL_ACTION_TO_KEYBOARD.items()
        }

        action["camera"] = self.last_mouse_delta
        self.last_mouse_delta = [0, 0]

        # FIXME: speical button toggle script mode
        if self.pressed_keys[key.P]:
            self.reset_angle = True
            self.handler.record_detect_animal = 'cow'
            print('reset angle')

        if self.pressed_keys[key.O]:
            self.run_script_action = True
            print('run script action')

        if self.pressed_keys[key.I]:
            self.pursue_mode = True
            print('pursue mode')

        if self.pressed_keys[key.U]:
            self.wait_animal_mode = True
            if self.handler.record_detect_animal is None:
                self.handler.record_detect_animal = input('detect animal: ')
            print('wait animal mode')

        if self.pressed_keys[key.Y]:
            self.plane_detect_mode = True
            print('plane detect mode')

        if self.pressed_keys[key.L]:
            self.plane_fill_mode = True
            print('plane fill mode')
        
        if self.pressed_keys[key.K]:
            self.save_video = True

        if self.pressed_keys[key.J]:
            self.cheat_animal = True

        if self.pressed_keys[key.M]:
            self.make_hole = True

        if self.reset_angle == True:
            action, done, _ = self.handler.reset_agent_pos_angle_action_v2(self.obs)
            if done:
                self.reset_angle = 2
                print('reset angle done')
        if self.reset_angle == 2:
            action, done = self.handler.reset_agent_pos_angle_action_v2_1(self.obs)
            if done:
                self.reset_angle = False
                self.plane_fill_mode = True

        
        if self.pursue_mode:
            yolo_res = self.animal_detector.inference(self.obs)
            handled_img, handled_yolo_res = self.handler.cv_handle(self.obs, yolo_res)
            if self.handler.detect_animal_flag:
                print('-------------------------------')
                action, done = self.handler.pursue_action(action, handled_yolo_res)
                if done:
                    self.pursue_mode = False
                    print('pursue mode done')

            cv2.imshow('yolo-res', handled_img)
            if cv2.waitKey(1) & 0XFF == 27:  # 退出键,  27=ESC
                return

        self.wait_animal_mode_count += 1
        if self.wait_animal_mode:
            if self.wait_animal_mode_count % 100 == 0:
                self.wait_animal_mode_count -= 1
                yolo_res = self.animal_detector.inference(self.obs)
                action, done = self.handler.load_back_wait_animal(yolo_res)
                if done:
                    self.wait_animal_mode_count += 1
                    self.wait_animal_mode = False
                    print('wait animal mode done')
        
        if self.plane_detect_mode:
            action, done = self.handler.multi_plane_detect(self.obs)
            if done:
                self.plane_detect_mode = False

        if self.cheat_animal:
            if self.handler.record_detect_animal is None:
                self.handler.record_detect_animal = input('detect animal: ')
            script_traj = []

            for _ in range(30):
                for _ in range(7):
                    script_traj.append([self.handler.animal_hotbar_dict[self.handler.record_detect_animal], 1])
                for _ in range(25):  # [ ,30]
                    script_traj.append(['hotbar.1', 1])
                print('----------------------------')

        if self.plane_fill_mode:
            action, done = self.handler.plane_detect_via_fill_v5_pre(self.obs)
            if done:
                self.plane_fill_mode = False
                self.make_hole = True

        if self.make_hole:
            action, done = self.handler.make_hole_v3(self.obs)
            if done:
                self.make_hole = False
                self.plane_fill_mode2 = True

        if self.plane_fill_mode2:
            action, done = self.handler.plane_detect_via_fill_v5(self.obs)
            if done:
                self.plane_fill_mode2 = False
                self.run_script_action = True

        script_traj = []
        # 0. 全局攻击
        script_traj.append(['camera', [-25, 0]])
        for _ in range(10):
            script_traj.append(['camera', [0, 0]])
        for _ in range(36):
            script_traj.append(['camera', [0, 10]])
            script_traj.append(['attack', 1])
            script_traj.append(['camera', [0, 0]])
        # 1. 切hotbar
        script_traj.append(['hotbar.1', 1])
        # 2.初始位置标定
        for _ in range(5):
            script_traj.append(['back', 1])
        for _ in range(5):
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['forward', 1, 'sneak', 1, 'use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(2):
            script_traj.append(['back', 1])
        # 3. 建一个篱笆
        for _ in range(5):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 4. 建门
        script_traj.append(['hotbar.2', 1])
        for _ in range(5):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 5. 建两个篱笆
        script_traj.append(['hotbar.1', 1])
        for _ in range(2):
            for _ in range(5):
                script_traj.append(['left', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 6. 后退3格，建下方篱笆，卡位
        for _ in range(3):
            for idx in range(5):
                script_traj.append(['back', 1])
                if idx >= 2:
                    script_traj.append(['attack', 1])
                    for _ in range(2):  # blank action
                        script_traj.append(['camera', [0, 0]])
                    script_traj.append(['use', 1])
        for _ in range(4):
            script_traj.append(['forward', 1])
        for _ in range(2):
            script_traj.append(['back', 1])
        # 7. 左转后退，建篱笆（4个）
        script_traj.append(['camera', [0, -90]])
        for _ in range(4):
            for idx in range(5):
                script_traj.append(['back', 1])
                if idx >= 2:
                    script_traj.append(['attack', 1])
                    for _ in range(2):  # blank action
                        script_traj.append(['camera', [0, 0]])
                    script_traj.append(['use', 1])
        for _ in range(4):
            script_traj.append(['forward', 1])
        for _ in range(2):
            script_traj.append(['back', 1])
        # 8. 左转后退，建篱笆（3个）
        script_traj.append(['camera', [0, -90]])
        for _ in range(3):
            for idx in range(5):
                script_traj.append(['back', 1])
                if idx >= 2:
                    script_traj.append(['attack', 1])
                    for _ in range(2):  # blank action
                        script_traj.append(['camera', [0, 0]])
                    script_traj.append(['use', 1])
        for _ in range(4):
            script_traj.append(['back', 1])
        for _ in range(3):
            script_traj.append(['forward', 1])
        # 9. 右走，左转，建封口篱笆
        for _ in range(3):
            script_traj.append(['right', 1])
        script_traj.append(['camera', [0, -90]])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 10. 后退，敲出动物出口
        for _ in range(5):
            script_traj.append(['back', 1])
        for _ in range(30):
            script_traj.append(['attack', 1])
        # 11. 出门
        script_traj.append(['camera', [0, -90]])
        for _ in range(4):
            script_traj.append(['forward', 1])
        for _ in range(2):
            script_traj.append(['back', 1])
        script_traj.append(['use', 1])
        for _ in range(6):
            script_traj.append(['forward', 1])
        # 10. 回头关门
        for _ in range(6):
            script_traj.append(['camera', [0, 30]])
        script_traj.append(['use', 1])
        # 11. 后退、抬头
        for _ in range(4):
            script_traj.append(['back', 1])
        script_traj.append(['camera', [-30, 0]])

        # -1.
        for _ in range(3):
            script_traj.append(['camera', [0, 0]])

        if self.run_script_action:
            action = self.env.action_space.noop()
            script_transition = script_traj[self.action_index]
            action[script_transition[0]] = script_transition[1]
            if len(script_transition) == 4:
                action[script_transition[2]] = script_transition[3]
            if len(script_transition) == 6:
                action[script_transition[2]] = script_transition[3]
                action[script_transition[4]] = script_transition[5]

            print(self.action_index, len(script_traj))
            self.action_index += 1
            if self.action_index == len(script_traj):
                self.action_index = 0
                self.run_script_action = False
                self.save_video = True

        if self.save_video:
            vout = cv2.VideoWriter()
            vout.open('./human_inference.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20, (640, 360), True)
            for img in self.obses:
                vout.write(img)
            vout.release()
            self.save_video = False

        #self.agent_map.step(action, self.location_stats, self.visual_direction)
        self.handler.step(action)
        # agent_traj, agent_traj_gt = self.agent_map.render(return_gt=True)

        # n_agent_traj = agent_traj.shape[0]
        # map_background = np.zeros((640, 640, 3), dtype=np.uint8)
        # for i in range(n_agent_traj-1):
        #     cv2.line(map_background, (int(agent_traj[i, 0]), int(agent_traj[i, 1])), (int(agent_traj[i+1, 0]), int(agent_traj[i+1, 1])), (0, 255, 0), 2)

        # n_agent_traj = agent_traj_gt.shape[0]
        # for i in range(n_agent_traj-1):
        #     cv2.line(map_background, (int(agent_traj_gt[i, 0]), int(agent_traj_gt[i, 1])), (int(agent_traj_gt[i+1, 0]), int(agent_traj_gt[i+1, 1])), (255, 0, 0), 2)
        # cv2.imshow('traj', map_background)

        #res, edges, _ = self.handler.is_plane_space(self.obs)
        #print(res)
        #cv2.imshow('canny', edges)
        #action['attack'] = 0

        #print(self.handler.is_plane_space_via_plane_detect_model(self.obs))
        print(self.handler.is_stone_space(self.obs))
        #print(self.handler.water_direction(self.obs))
        return action

    def _show_message(self, text):
        label = pyglet.text.Label(
            text,
            font_size=32,
            x=self.window.width // 2,
            y=self.window.height // 2,
            anchor_x='center',
            anchor_y='center'
        )
        label.draw()
        self.window.flip()

    def reset(self):
        self.window.clear()
        self._show_message("Resetting environment...")
        obs = self.env.reset()
        self.obs = obs["pov"]
        self.location_stats = obs["location_stats"]
        self._update_image(obs["pov"])
        self.clock.tick()

        # FIXME: script mode
        self.script_mode = False
        #self.agent_map.reset(obs['location_stats'])
        #self.of.reset(obs['pov'])
        self.handler.reset(obs['pov'])
        return obs

    def step(self, action: Optional[dict] = None):
        """
        Step environment for one frame.

        If `action` is not None, assume it is a valid action and pass it to the environment.
        Otherwise read action from player (current keyboard/mouse state).

        If `override_if_human_input` is True, execeute action from the human player if they
        press any button or move mouse.

        The executed action will be added to the info dict as "taken_action".
        """

        # FIXME: override action
        override_action = {}
        if action:
             for key in action:
                if key.startswith('hotbar'):
                    override_action[key] = action[key]

        # human action
        time_to_sleep = MINERL_FRAME_TIME - self.clock.tick()
        if time_to_sleep > 0:
            self.clock.sleep(int(time_to_sleep * 1000))

        self.window.dispatch_events()
        human_action = self._get_human_action()
        action = human_action
        
        # FIXME: override action
        if len(override_action) > 0:
            for key in override_action:
                action[key] = override_action[key]
        
        obs, reward, done, info = self.env.step(action)
        self.obs = obs["pov"]
        self.location_stats = obs["location_stats"]

        # yolo_res = self.animal_detector.inference(obs["pov"])
        # handled_img, handled_yolo_res = self.handler.cv_handle(obs['pov'], yolo_res)
        self.obses.append(obs["pov"].copy())
        # cv2.imshow(' yolo', handled_img)
        self._update_image(obs["pov"])

        # of
        # _, diffs, ori_corner_poses = self.of.step(self.obs)
        # visual_direction = self.of.handle_direction(diffs, ori_corner_poses)
        # self.visual_direction = self.of.direction_filter(visual_direction, action)

        if done:
            self._show_message("Episode done.")

        info["taken_action"] = action
        return obs, reward, done, info
