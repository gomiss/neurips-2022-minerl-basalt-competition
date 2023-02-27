import argparse
import copy
import os
import platform
import queue
import sys
from pathlib import Path
from queue import Queue

from utils.torch_utils import select_device, smart_inference_mode
from FindCaveEnvWarper import FindCaveEnvWarper
from DetectCave import DetectCave
from argparse import ArgumentParser
import pickle
import math
import cv2
import numpy as np
from agent import MineRLAgent, ENV_KWARGS
from enum import Enum
from config import STATE, ACTION
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from config import IMG_WIDTH


def rotateToPos(pos, env):
    rotateAngle = (pos / IMG_WIDTH - 0.5) * 51.2 * 2 / 6 # (pos-IMG_WIDTH/2) / IMG_WIDTH * 51.2 * 2 / 6
    print(rotateAngle)
    minerl_action = env.action_space.sample()
    list_key = list(minerl_action.keys())

    value = [np.array(0) for i in range(24)]
    value[3] = np.array([0, rotateAngle])
    for j, kk in enumerate(list_key):
        minerl_action[kk] = value[j]

    return minerl_action


def area_of_rect(leftup, rightdown):
    assert rightdown[0] - leftup[0] > 0
    assert rightdown[1] - leftup[1] > 0

    return (rightdown[0] - leftup[0]) * (rightdown[1] - leftup[1])


def minerl_go_action(game_env):
    go = game_env.action_space.sample()
    list_key = list(go.keys())
    value = [np.array(0) for j in range(24)]
    value[3] = np.array([0, 0])
    for j, kk in enumerate(list_key):
        go[kk] = value[j]
    go['forward'] = np.array(1)
    return go


def minerl_gojump_action(game_env):
    gojump = game_env.action_space.sample()
    list_key = list(gojump.keys())
    value = [np.array(0) for j in range(24)]
    value[3] = np.array([0, 0])
    for j, kk in enumerate(list_key):
        gojump[kk] = value[j]
    gojump['forward'] = np.array(1)
    gojump['jump'] = np.array(1)
    return gojump


def minerl_terminal_action(game_env):
    terminal = game_env.action_space.sample()
    list_key = list(terminal.keys())
    value = [np.array(0) for j in range(24)]
    value[3] = np.array([0, 0])
    for j, kk in enumerate(list_key):
        terminal[kk] = value[j]
    terminal['ESC'] = np.array(1)
    return terminal


if __name__ == "__main__":
    yolo_weights = os.path.join(ROOT, r'runs\train\exp12\weights\best.pt')
    env = FindCaveEnvWarper("MineRLBasaltFindCave-v0")
    obs = env.reset()
    device = select_device('cuda:0')
    detect_cave = DetectCave(device,yolo_weights)

    print("---Loading model---")

    agent_parameters = pickle.load(open('../train/foundation-model-1x.model', "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env.env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(r'../train/MineRLBasaltFindCave.weights')

    ROTATE_ACTION_LIST = [ACTION.ROTATE for i in range(4)]
    go = minerl_go_action(env)
    gojump = minerl_gojump_action(env)
    terminal = minerl_terminal_action(env)
    state = STATE.VPT_WALKING
    step = 0
    previous10_img = queue.Queue(10)
    pre_state = STATE.VPT_WALKING
    done = False
    while True:
        img = obs['pov']
        # cv2.imwrite("previous5/" + str(step) + ".png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print("state:", state, pre_state, step)

        if not previous10_img.full():
            previous10_img.put(img)
        else:
            previous10_img.get()
            previous10_img.put(img)
            if step % 100:
                diff_count = 0
                if state == STATE.GO_TO_CAVE or state == STATE.VPT_WALKING:
                    while not previous10_img.empty():
                        prev = previous10_img.get()
                        difference = cv2.subtract(prev, img)
                        if abs(np.average(difference)) < 2:
                            diff_count += 1
                            print("stuckstuckstuckstuckstuckstuckstuckstuck")
                            # break
                    if diff_count > 2:
                        env.is_stuck = True

        # print(img.shape
        if state == STATE.VPT_WALKING:
            all_pos_set, agent_in_cave = detect_cave.detect_image(img, device)
            if agent_in_cave:
                obs, reward, done, info = env.step(terminal)
            else:
                if 'cave' in all_pos_set:
                    pos_set = all_pos_set['cave']
                elif 'hole' in all_pos_set:
                    pos_set = all_pos_set['hole']
                else:
                    pos_set = []
                if len(pos_set) > 0:
                    pre_state = state
                    state = STATE.FIND_CAVE_AND_ROTATE
                    ROTATE_ACTION_IDX = 0
                    # continue
                else:
                    minerl_action = agent.get_action(obs)
                    minerl_action["ESC"] = 0
                    minerl_action['drop'] = np.array([0])
                    minerl_action['inventory'] = np.array([0])
                    minerl_action['use'] = np.array([0])
                    minerl_action['sneak'] = np.array([0])
                    # minerl_action['attack'] = np.array([0])
                    for num in range(1,10):
                        minerl_action['hotbar.{}'.format(num)] = np.array([0])
                    minerl_action['camera'] = minerl_action['camera'][0]
                    obs, reward, done, info = env.step(minerl_action, True)
                    pre_state = state
        elif state == STATE.FIND_CAVE_AND_ROTATE:
            pos = (pos_set[0][0] + pos_set[0][2]) / 2
            if ROTATE_ACTION_IDX == 0:
                pre_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if ROTATE_ACTION_IDX > len(ROTATE_ACTION_LIST) or math.fabs(pos - img.shape[1]/2) < 10:
                pre_state = state
                state = STATE.GO_TO_CAVE
            else:
                ROTATE_ACTION_IDX += 1

                rect_left_up = (pos_set[0][0], pos_set[0][1])
                rect_right_down = (pos_set[0][2], pos_set[0][3])
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(pre_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                new_rect_left_up = rect_left_up + flow[rect_left_up[1], rect_left_up[0]]
                new_rect_right_down = rect_right_down + flow[rect_right_down[1] - 1, rect_right_down[0] - 1]
                rect_left_up = (np.clip(new_rect_left_up[0], 0, flow.shape[1] - 1),
                                np.clip(new_rect_left_up[1], 0, flow.shape[0] - 1))
                rect_right_down = (np.clip(new_rect_right_down[0], 0, flow.shape[1] - 1),
                                   np.clip(new_rect_right_down[1], 0, flow.shape[0] - 1))

                rect_left_up = tuple(np.int32(rect_left_up))
                rect_right_down = tuple(np.int32(rect_right_down))
                show_img = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2RGB)
                cv2.rectangle(show_img, rect_left_up, rect_right_down, (0, 0, 255), 3)
                cv2.imshow('yolo2', show_img)

                minerl_action = rotateToPos(pos, env)
                print("myrotae: ",  minerl_action)
                print("i=", ROTATE_ACTION_IDX)
                obs, reward, done, info = env.step(minerl_action, False)
                print("action END")
                env.render()
                print("render END")
                pre_gray = gray
                img = obs['pov']
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif state == STATE.GO_TO_CAVE:
            # first time enter GOTOCAVE
            if pre_state == STATE.FIND_CAVE_AND_ROTATE:
                all_pos_set,agent_in_cave = detect_cave.detect_image(img, device)
                if agent_in_cave:
                    obs, reward, done, info = env.step(terminal)
                else:
                    if 'cave' in all_pos_set:
                        pos_set = all_pos_set['cave']
                    elif 'hole' in all_pos_set:
                        pos_set = all_pos_set['hole']
                    # else:
                    #     pos_set = []
                    if len(pos_set) > 0:
                        rect_left_up = (pos_set[0][0], pos_set[0][1])
                        rect_right_down = (pos_set[0][2], pos_set[0][3])

                        center = ((rect_left_up[0] + rect_right_down[0]) / 2, (rect_left_up[1] + rect_right_down[1]) / 2)
                        # rect above
                    else:
                        print("no centetnocenter, no centetnocenterno centetnocenterno centetnocenterno centetnocenter")
                        # center = (img.shape[1]/2,img.shape[0]/2,)
                    next_action = Queue()

                    if center[1]/img.shape[0] <= 1/2:
                        for n in range(3):
                            next_action.put(copy.deepcopy(gojump))
                        for n in range(2):
                            next_action.put(copy.deepcopy(go))
                        # next_action.put(copy.deepcopy(go))

                    elif 1/2 < center[1] / img.shape[0] < 3/4:
                        next_action.put(copy.deepcopy(gojump))
                        for n in range(4):
                            next_action.put(copy.deepcopy(go))
                        # next_action.put(copy.deepcopy(go))
                        # next_action.put(copy.deepcopy(go))
                        # next_action.put(copy.deepcopy(go))

                    elif center[1] / img.shape[0] >= 3/4:
                        for n in range(5):
                            next_action.put(copy.deepcopy(go))
                    print(next_action.empty())
                    pre_state = state
                    state = STATE.GO_TO_CAVE
                    contius_not_found = 0

            elif pre_state == STATE.GO_TO_CAVE:
                if not next_action.empty():
                    now_action = next_action.get()
                    obs, reward, done, info = env.step(now_action, False)
                all_pos_set, agent_in_cave = detect_cave.detect_image(img, device)
                if agent_in_cave:
                    obs, reward, done, info = env.step(terminal)
                else:
                    if 'cave' in all_pos_set:
                        pos_set = all_pos_set['cave']
                    elif 'hole' in all_pos_set:
                        pos_set = all_pos_set['hole']
                    else:
                        pos_set = []
                    if len(pos_set) > 0:
                        pre_state = state
                        state = STATE.FIND_CAVE_AND_ROTATE
                        ROTATE_ACTION_IDX = 0
                        contius_not_found = 0
                    else:
                        contius_not_found += 1

                        if next_action.empty():
                            if contius_not_found <= 10:
                                rotateAngle1 = (center[1] - 160) / 320 * 51.2 * 2 / 10
                                rotateAngle2 = (center[0] - 320) / 640 * 51.2 * 2 / 10
                                minerl_action = env.action_space.sample()
                                list_key = list(minerl_action.keys())

                                value = [np.array(0) for i in range(24)]

                                value[3] = np.array([rotateAngle1, rotateAngle2])
                                print('rotate: ', value[3])
                                for j, kk in enumerate(list_key):
                                    minerl_action[kk] = value[j]
                                next_action.put(minerl_action)
                                print('putputputputput', rotateAngle1, rotateAngle2)
                            else:
                                pre_state = state
                                state = STATE.VPT_WALKING
                        print("contiusnot = ", contius_not_found)

        # pre_state = state
        step+=1
        if done:
            obs = env.reset()
            detect_cave.reset()
            state = STATE.VPT_WALKING
            pre_state = STATE.VPT_WALKING


