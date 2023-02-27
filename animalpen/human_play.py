from copy import deepcopy
from collections import Counter

import gym
import minerl
import cv2
import numpy as np

import matplotlib.pyplot as plt

from custom_human_play_interface import HumanPlayInterface
from visual_direction import OptialFlow

IS_OPTIAL_FLOW = False
IS_CONNECT_COMPONENT = False
IS_CANNY = False


def main():
    if IS_OPTIAL_FLOW:
        of = OptialFlow()

    env = gym.make('MineRLBasaltCreateVillageAnimalPen-v0')
    env = HumanPlayInterface(env)
    print("---Launching MineRL enviroment (be patient)---")
    
    obs = env.reset()
    if IS_OPTIAL_FLOW:
        of.reset(obs['pov'])

    while True:
        obs, _, done, _ = env.step()


        if IS_CONNECT_COMPONENT:
            img = obs['pov']
            img = cv2.medianBlur(img, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # 阈值分割得到二值化图片
            ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            #ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            # 膨胀操作
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            bin_clo = cv2.dilate(binary, kernel2, iterations=2)
            # 连通域分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)
            # 不同的连通域赋予不同的颜色
            output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
            for i in range(1, num_labels):
                mask = labels == i
                output[:, :, 0][mask] = np.random.randint(0, 255)
                output[:, :, 1][mask] = np.random.randint(0, 255)
                output[:, :, 2][mask] = np.random.randint(0, 255)
            cv2.imshow('bin_img', bin_clo)
            cv2.imshow('connect_component', output)
            if cv2.waitKey(1) & 0XFF == 27:  # 退出键,  27=ESC
                break

        if IS_OPTIAL_FLOW:
            img, diffs, ori_corner_poses = of.step(obs['pov'])
            visual_direction = of.handle_direction(diffs, ori_corner_poses, show=True)
            visual_direction = of.direction_filter(visual_direction)
            cv2.imshow('optial_flow', img)
            if cv2.waitKey(1) & 0XFF == 27:  # 退出键,  27=ESC
                break
        
        if IS_CANNY:
            img = obs['pov']
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 均值滤波
            img_gray_blur = cv2.bilateralFilter(img_gray, 7, 25, 25) # 9 75 75
            edges = cv2.Canny(img_gray_blur, 50, 150)

            cv2.imshow('gray', img_gray_blur)
            cv2.imshow('canny', edges)
            if cv2.waitKey(1) & 0XFF == 27:  # 退出键,  27=ESC
                break

        if done:
            env.reset()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


