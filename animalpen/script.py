from copy import deepcopy
from collections import Counter
import jsonlines
import random
import os

import cv2
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms

from animalpen.visual_direction import OptialFlow


KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}

# Template action
NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}

MINEREC_ORIGINAL_HEIGHT_PX = 720
# Matches a number in the MineRL Java code
# search the code Java code for "constructMouseState"
# to find explanations
CAMERA_SCALER = 360.0 / 2400.0

# If GUI is open, mouse dx/dy need also be adjusted with these scalers.
# If data version is not present, assume it is 1.
MINEREC_VERSION_SPECIFIC_SCALERS = {
    "5.7": 0.5,
    "5.8": 0.5,
    "6.7": 2.0,
    "6.8": 2.0,
    "6.9": 2.0,
}

# =============================
# hyperparameters
CV_HANDLER_TOP_THRESHOLD = 50
CV_HANDLER_LEFT_RIGHT_THRESHOLD = 30
CV_HANDLER_MIN_DETECT_THRESHOLD = 10

MEAN_ANIMAL_WINDOW_SIZE = {
    'rabbit': 2500,
    'pig'   : 6000,
    'duck'  : 4000,
    'cow'   : 8000,
    'sheep' : 8000
}

CONF_ANIMAL = {
    'rabbit': 0.5,
    'pig'   : 0.7,
    'duck'  : 0.7,
    'cow'   : 0.7,
    'sheep' : 0.7
}

LOOK_DOWN_ANGLE = 40
LOOK_UP_ANGLE = 40
# =============================

class Handler():
    MIDDLE_X = 320
    MIDDLE_Y = 160

    def __init__(self, env, path, max_frame=3, jsonfile_path='./resources/action_seq.jsonl'):
        self.env = env
        self.dir = path
        self.max_frame = max_frame
        self.jsonfile_path = jsonfile_path

        # 光流检测->平面检测
        self.of = OptialFlow()
        self.plane_calibration_index = 0 # 平面校准动作索引
        self.plane_calibration_back_index = 0
        self.plane_calibration_back_index_max = 70
        self.plane_calibration_init_count = 0
        self.last_img = None

        self.multi_plane_detect_status = 0
        self.multi_plane_detect_index = 0 # 平面校准动作索引

        # animal_hotbar dict
        self.animal_hotbar_dict = {
            'rabbit': 'hotbar.3',
            'pig'   : 'hotbar.3',
            'duck'  : 'hotbar.4',
            'cow'   : 'hotbar.5',
            'sheep' : 'hotbar.5'
        }

        # 标定->姿态角
        self.angle_x = 0
        self.angle_y = 0
        self.target_angle = [90, 0]

        # 标定->中心点坐标
        self.center_x = None
        self.center_y = None

        # yolo检测信息记录
        self.frame_info = []
        self.detect_animal_flag_info = []
        
        # yolo检测，干预闲逛模型计数
        self.mask_with_yolo_count = 0
        self.max_mask_with_yolo_count = 3
        self.record_mask_with_yolo_x = self.MIDDLE_X 
        
        # angle pos重置计数
        self.reset_angle_count = 0
        
        # 追动物计数
        self.pursue_count = 0
        self.max_pursue_count = 10
        self.record_animal_center_pos_x  = self.MIDDLE_X 
        self.has_reset_angle = False

        # 等动物计数（等待动物走过来）
        self.wait_count = 0
        self.max_wait_count = 40

        # 检测到的动物
        self.record_detect_animal = None

        # 坐标标定flag
        self.reset_agent_angle_action_flag = 0
        self.reset_agent_pos_angle_action_flag = 0
        self.reset_agent_pos_angle_action_v2_flag = 0

        # 回头看动作计数
        self.load_back_wait_animal_index = 0

        # 收集物品动作计数
        self.collect_item_index = 0
        self.collect_item_check_flag = None
        self.collect_item_check_index = 0

        # 空地标记
        self.plane_space_list = []
        self.max_plane_space_list = 5

        # 填空地动作计数
        self.plane_detect_via_fill_index = 0

        self.plane_detect_via_fill_status = 0 # v2: 0; v3: -1        
        self.plane_detect_via_fill_block_status = 0
        self.plane_detect_via_fill_block_turn_status = 0
        self.plane_detect_via_fill_last_img = None
        self.plane_detect_fill_or_dig_v2_1_count = 0
        self.plane_detect_fill_or_dig_count = 0
        self.plane_detect_via_fill_v5_pre_index = 0

        # 重检测空地动作计数
        self.recheck_plane_space_index = 0

        # 围动物，脚本指令计数
        self.pen_action_index = 0
        self.pen_traj = self.get_pen_traj_v6()
        self.pen_traj_rabbit = self.get_pen_traj_v4_3()

        # 脱困动作计数
        self.stuck_action_index = 0

        self.video_record = []
        self.img_list = []
        self.max_img_count = 10
        self.step_cnt = 0

        # 遇到水转
        self.water_direction_count = 0

        # 建洞，困动物
        self.make_hole_index = 0
        self.pre_make_hole = None
        self.is_stone_for_make_hole = None

        # 保底策略计数
        self.minimum_guarantee_index = 0

        # 被困图像list
        self.trapped_img_list = []
        self.trapped_img_list_max = 10

        # 初始化平地检测模型
        PLANE_DETECT_MODEL_PATH = os.path.join(self.dir, 'train', 'animal_pen_flattenareaV4resnet50.pt')
        self.init_plane_detect_model(PLANE_DETECT_MODEL_PATH)
        self.block1 = cv2.imread(
            os.path.join(self.dir, 'hub', 'checkpoints', 'animal_pen_block_1.png')
        )
        self.block2 = cv2.imread(
            os.path.join(self.dir, 'hub', 'checkpoints', 'animal_pen_block_2.png')
        )

        # 潜水检测模型
        WATER_DETECT_TEMPLATE_PATH = './animalpen/resources/bubble_template.png'
        WATER_DETECT_MASK_PATH = './animalpen/resources/bubble_mask.png'
        self.water_detect_template = cv2.imread(WATER_DETECT_TEMPLATE_PATH, 0)
        self.water_detect_mask = cv2.imread(WATER_DETECT_MASK_PATH, 0)

    def reset(self, img):
        self.step_cnt = 0
        self.of.reset(img)
        self.img_list.append(img.copy())

    def step(self, action, img):
        self.angle_x += action['camera'][0]
        self.angle_y += action['camera'][1]

        if self.angle_x > 90:
            self.angle_x = 90
        if self.angle_x < -90:
            self.angle_x = -90
        if self.angle_y > 180:
            self.angle_y -= 360
        if self.angle_y < -180:
            self.angle_y += 360

        self.step_cnt += 1
        self.img_list.append(img.copy())
        if len(self.img_list) > self.max_img_count:
            self.img_list.pop(0)

    # ================================
    #   Imgs Script
    # ================================
    def cv_handle(self, img, yolo_res):
        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)

        handled_yolo_res = {}
        for key in yolo_res:
            _pos = yolo_res[key]['pos']
            _size = yolo_res[key]['size']
            _conf = yolo_res[key]['conf']
            _class = yolo_res[key]['class']

            # 1. 除去过高的识别（基本是把天空识别成了某种动物）
            pos_top = _pos[1] - _size[1] / 2
            if pos_top < CV_HANDLER_TOP_THRESHOLD:
                continue
            # 2. 除去边缘的位置的识别（防止把边界上的部分东西识别错误）
            if _pos[0] < CV_HANDLER_LEFT_RIGHT_THRESHOLD or _pos[0] > (640 - CV_HANDLER_LEFT_RIGHT_THRESHOLD):
                continue
            # 3. 除去过小的识别
            if _size[0] * _size[1] < CV_HANDLER_MIN_DETECT_THRESHOLD:
                continue
            # 4. 除去花朵和人
            if _class == 'flower' or _class == 'people':
                continue
            # 5. 基于conf优化
            if _conf < CONF_ANIMAL[_class]:
                continue
            
            pos_top_left = (int(_pos[0] - _size[0] / 2), int(_pos[1] - _size[1] / 2))
            pos_bottom_right = (int(_pos[0] + _size[0] / 2), int(_pos[1] + _size[1] / 2))
            cv2.rectangle(img, pos_top_left, pos_bottom_right, (255, 255, 0), 3)

            handled_yolo_res[key] = {
                'pos': _pos,
                'size': _size,
                'conf': _conf,
                'class': _class
            }
        
        self.frame_info.append(handled_yolo_res)
        if len(self.frame_info) > self.max_frame:
            self.frame_info.pop(0)
        
        try:
            img = img[:, :, ::-1]
            img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_LINEAR)
            self.video_record.append(img.copy())
        except:
            pass
        return img, handled_yolo_res
    
    def get_center_pos(self, img):
        # 灰度处理 & 双边滤波 & 边缘检测
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_blur = cv2.bilateralFilter(img_gray, 7, 25, 25)
        edges = cv2.Canny(img_gray_blur, 50, 150)

        # 霍夫变换
        lines = cv2.HoughLines(edges[:300, :], 1, np.pi/180, 60) 

        # 坐标提取
        poses_x = []
        poses_y = []
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a*rho, b*rho
                x1, y1 = int(x0 + 1000*(-b)), int(y0 + 1000*(a))
                if (abs(theta) < 0.05 or abs(theta-2 * np.pi) < 0.05 ):
                    poses_x.append((x0 + x1) / 2) 
                elif abs(theta - np.pi / 2) < 0.05:
                    poses_y.append((y0 + y1) / 2)
        
        poses_x.sort()
        poses_y.sort()

        center_poses_x = []
        center_poses_y = []
        for i in range(len(poses_x) - 1):
            for j in range(i):
                if j < i and 100 < (poses_x[i] - poses_x[j]) < 200:
                    center_poses_x.append((poses_x[i] + poses_x[j]) / 2)

        for i in range(len(poses_y) - 1):
            for j in range(i):
                if j < i and 100 < (poses_y[i] - poses_y[j]) < 200:
                    center_poses_y.append((poses_y[i] + poses_y[j]) / 2)

        # 中心点标定
        if len(center_poses_x) > 0:
            if self.center_x is None:
                self.center_x = np.median(center_poses_x)
            else:
                for center_pos_x in center_poses_x:
                    min_dist = float('inf')
                    if abs(center_pos_x - self.center_x) < min_dist:
                        self.center_x = center_pos_x
                        min_dist = abs(center_pos_x - self.center_x)
                self.center_x = 0.5 * self.center_x + 0.5 * np.median(center_poses_x)

        if len(center_poses_y) > 0:
            if self.center_y is None:
                self.center_y = np.median(center_poses_y)
            else:
                for center_pos_y in center_poses_y:
                    min_dist = float('inf')
                    if abs(center_pos_y - self.center_y) < min_dist:
                        self.center_y = center_pos_y
                        min_dist = abs(center_pos_y - self.center_y)
                self.center_y = 0.5 * self.center_y + 0.5 * np.median(center_poses_y)

        # if self.center_x is not None and self.center_y is not None:
        #     handled_img = img.copy()
        #     ptStart = (int(self.center_x), 1)
        #     ptEnd = (int(self.center_x), 300)
        #     point_color = (255, 255, 0)  # BGR
        #     thickness = 2
        #     lineType = 4
        #     cv2.line(handled_img, ptStart, ptEnd, point_color, thickness, lineType)

        #     ptStart = (1, int(self.center_y))
        #     ptEnd = (600, int(self.center_y))
        #     point_color = (255, 255, 0)  # BGR
        #     thickness = 2
        #     lineType = 4
        #     cv2.line(handled_img, ptStart, ptEnd, point_color, thickness, lineType)

        #     cv2.imshow('dsadas', handled_img)

        if self.center_x is not None and self.center_y is not None:
            return [self.center_x, self.center_y]
        return None

    def get_angle(self, img):
        # 灰度处理 & 双边滤波 & 边缘检测
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_blur = cv2.bilateralFilter(img_gray[30:300, :], 7, 25, 25)
        #edges = cv2.Canny(img_gray_blur, 50, 150)
        edges = cv2.Canny(img_gray_blur, 30, 150)

        # 霍夫变换 
        lines = cv2.HoughLines(edges, 1, np.pi/180, 70)

        # 角度扇区划分
        N_SECTOR = 10
        sector_theta = { i: [] for i in range(N_SECTOR) }
        if lines is not None:
            for line in lines:
                rho, theta = line[0]

                if np.pi / 4 < theta < 3 * np.pi / 4:
                    sector_theta[(theta - np.pi/4) // (np.pi / 2 / N_SECTOR)].append(theta)
                elif 0 < theta < np.pi / 4:
                    theta += np.pi / 2
                    sector_theta[(theta - np.pi/4) // (np.pi / 2 / N_SECTOR)].append(theta)
                elif 3 * np.pi / 4 < theta < np.pi:
                    theta -= np.pi / 2
                    sector_theta[(theta - np.pi/4) // (np.pi / 2 / N_SECTOR)].append(theta) 

                # a = np.cos(theta)
                # b = np.sin(theta)
                # x0 = a * rho
                # y0 = b * rho

                # x1 = int(x0 + 2000 * (-b))
                # y1 = int(y0 + 2000 * (a))
                # x2 = int(x0 - 2000 * (-b))
                # y2 = int(y0 - 2000 * (a))

                # cv2.line(img_gray_blur, (x1, y1), (x2, y2), (255, 0, 255), 1)

        max_count = 0
        sector_index = -1
        for sector_theta_key in sector_theta:
            if len(sector_theta[sector_theta_key]) > max_count:
                max_count = len(sector_theta[sector_theta_key])
                sector_index = sector_theta_key
        
        turn_angle = 0
        if sector_index != -1:
            turn_angle = (sum(sector_theta[sector_index]) / len(sector_theta[sector_index]) - np.pi / 2) * 180 / np.pi

        # print('turn_angle: ', turn_angle)
        # cv2.imshow('angle visualization', edges)
        # cv2.imshow('img gray', img_gray_blur)
        return turn_angle

    def is_plane_space(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # # 均值滤波
        #img_gray_blur = cv2.blur(img_gray, (3, 3))
        # 双边滤波
        img_gray_blur = cv2.bilateralFilter(img_gray, 7, 25, 25) # 9 75 75

        edges = cv2.Canny(img_gray_blur, 50, 150)
        total_edge_num = Counter(edges.flatten())[255]
        selected_area_edges_num = Counter(np.array([v[180:460] for v in edges[180:300]]).flatten())[255]

        if (selected_area_edges_num > 200 and selected_area_edges_num < 1000 and total_edge_num > 7000) or selected_area_edges_num/total_edge_num < 0.08:
            return True#, edges, img_gray_blur
        return False#, edges, img_gray_blur

    def recheck_plane_space(self, img):
        self._get_target_y_angle()

        script_traj = []
        script_traj.append(['camera', [70 - self.angle_x, self.target_angle[1] - self.angle_y]])

        action = deepcopy(NOOP_ACTION)
        script_action = script_traj[self.recheck_plane_space_index]
        action[script_action[0]] = script_action[1]

        self.recheck_plane_space_index += 1
        if self.recheck_plane_space_index >= len(script_traj):
            self.recheck_plane_space_index = 0
            res = self.is_plane_space(img) and self.is_plane_space_via_plane_detect_model(img)
            return action, res
        return action, -1

    def recheck_plane_space_water_limit(self, img):
        self._get_target_y_angle()

        script_traj = []
        script_traj.append(['camera', [70 - self.angle_x, self.target_angle[1] - self.angle_y]])

        action = deepcopy(NOOP_ACTION)
        script_action = script_traj[self.recheck_plane_space_index]
        action[script_action[0]] = script_action[1]

        self.recheck_plane_space_index += 1
        if self.recheck_plane_space_index >= len(script_traj):
            self.recheck_plane_space_index = 0
            if self.is_water_space(img):
                return action, False
            return action, self.is_plane_space(img)
        return action, -1

    def init_plane_detect_model(self, path):
        # self.plane_detect_model = models.resnet50()

        # num_ftrs = self.plane_detect_model.fc.in_features
        # self.plane_detect_model.fc = nn.Linear(num_ftrs, 2)
        # self.plane_detect_model.load_state_dict(torch.load(path))

        self.plane_detect_model = torch.load(path)
        self.plane_detect_model.eval()

    def is_plane_space_via_plane_detect_model(self, img):
        if self.is_water_space(img, 0.02):
            return False

        img_transforms = transforms.Compose([
            transforms.ToTensor(),    
            transforms.Resize((224,224)),                            
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],),
        ])

        img = img_transforms(img.copy())
        img = img.unsqueeze(0).cuda()
        is_plane_space_value = torch.sigmoid(self.plane_detect_model(img)).cpu().detach().numpy()[0][0]
        
        return is_plane_space_value > 0.2

    def is_water_space(self, img, limit=0.08):
        lower = np.array([100, 43, 46])
        upper = np.array([124, 255, 255])
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        bluemask = cv2.inRange(hsv, lower, upper)[130:300, 100:540] / 255. 
        if bluemask.mean() > limit:
            return True
        else:
            return False

    def is_lava_space(self, img):
        lower = np.array([0, 43, 46])
        upper = np.array([10, 255, 255])
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        redmask1 = cv2.inRange(hsv, lower, upper)[100:300, 100:540] / 255.

        lower = np.array([156, 43, 46])
        upper = np.array([180, 255, 255])
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        redmask2 = cv2.inRange(hsv, lower, upper)[100:300, 100:540] / 255.

        redmask = (redmask1 + redmask2) > 1
        if redmask.mean() > 0.2:
            return True
        else:
            return False

    def is_snow_space(self, img):
        lower = np.array([0, 0, 221])
        upper = np.array([180, 30, 255])
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        bluemask = cv2.inRange(hsv, lower, upper)[120:240, 256:384] / 255. 
        if bluemask.mean() > 0.5:
            return True
        else:
            return False

    def is_stone_space(self, img):
        lower = np.array([23, 43, 46])
        upper = np.array([34, 255, 255])
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        bluemask = cv2.inRange(hsv, lower, upper)[120:240, 256:384] / 255. 
        if bluemask.mean() > 0.5:
            return True
        else:
            return False
        
    def water_direction(self, img):
        water_img = img.copy()
        lower = np.array([100, 43, 46])
        upper = np.array([124, 255, 255])
        hsv = cv2.cvtColor(water_img, cv2.COLOR_RGB2HSV)
        bluemask = cv2.inRange(hsv, lower, upper)[180:300, 100:540] / 255. 
        
        left_blue = bluemask[:, :320].mean()
        right_blue = bluemask[:, 320:].mean()

        if self.water_direction_count > 10:
            self.water_direction_count -= 1
            return 0

        if left_blue > 0.3 and right_blue > 0.3:
            self.water_direction_count += 10
            return 20
        elif left_blue > 0.3:
            self.water_direction_count += 1
            return 10
        elif right_blue > 0.3:
            self.water_direction_count += 1
            return -10
        else:
            self.water_direction_count = 0
            return 0
        
    def water_direction_limit(self, img):
        water_img = img.copy()
        lower = np.array([100, 43, 46])
        upper = np.array([124, 255, 255])
        hsv = cv2.cvtColor(water_img, cv2.COLOR_RGB2HSV)
        bluemask = cv2.inRange(hsv, lower, upper)[180:300, 100:540] / 255. 
        
        left_blue = bluemask[:, :320].mean()
        right_blue = bluemask[:, 320:].mean()

        if self.water_direction_count > 10:
            self.water_direction_count -= 1
            return 0

        LIMIT = 0.95
        if left_blue > LIMIT and right_blue > LIMIT:
            self.water_direction_count += 10
            return 180
        elif left_blue > LIMIT:
            self.water_direction_count += 1
            return 10
        elif right_blue > LIMIT:
            self.water_direction_count += 1
            return -10
        else:
            self.water_direction_count = 0
            return 0

    def block_similarity(self, img):
        block_img = img[340:360, 330:348]

        block_1_similarity = (block_img - self.block1).mean()
        block_2_similarity = (block_img - self.block2).mean()
        block_similarity = block_1_similarity if block_1_similarity < block_2_similarity else block_2_similarity

        return block_similarity / 255.

    def is_stuck(self):
        def _frame_diff(img1, img2):
            difference = cv2.subtract(img1, img2)
            return abs(np.average(difference))

        if len(self.img_list) != self.max_img_count:
            return False

        for obs in self.img_list[:-1]:
            if _frame_diff(obs, self.img_list[-1]) < 2:
                return True
        
        return False

    def is_in_water(self, img):
        THRESHOLD = 0.6

        img = img[311:320, 402:411]
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        res = cv2.matchTemplate(img_gray, self.water_detect_template, cv2.TM_CCOEFF_NORMED, mask=self.water_detect_mask)

        loc = np.where(res >= THRESHOLD)
        if len(loc[0]) > 0:
            return True
        return False

    # ================================
    #   Action Script
    # ================================#
    def collect_item(self):
        self._get_target_y_angle()

        script_traj = []
        script_traj.append(['camera', [50 - self.angle_x, self.target_angle[1] - self.angle_y]])
        for _ in range(3):
            for _ in range(3):
                for _ in range(30): 
                    script_traj.append(['attack', 1])
                for _ in range(10):
                    script_traj.append(['forward', 1])

            script_traj.append(['camera', [0, 90]])
        script_traj.append(['camera', [-50, 0]])

        action = deepcopy(NOOP_ACTION)
        script_action = script_traj[self.collect_item_index]
        action[script_action[0]] = script_action[1]
        
        self.collect_item_index += 1
        if self.collect_item_index >= len(script_traj):
            self.collect_item_index = 0
            return action, True
        return action, False

    def collect_item_check(self, img):
        if self.collect_item_check_flag is None:
            # print(self.block_similarity(img))
            self.collect_item_check_flag = self.block_similarity(img) < 0.3
        
        if self.collect_item_check_flag:
            self.collect_item_check_flag = None
            action = deepcopy(NOOP_ACTION)
            return action, True
        else:
            script_traj = []
            script_traj.append(['hotbar.6', 1])
            script_traj.append(['swapHands', 1])
            script_traj.append(['hotbar.8', 1])
            script_traj.append(['swapHands', 1])
            script_traj.append(['hotbar.7', 1])
            script_traj.append(['swapHands', 1])
            script_traj.append(['hotbar.6', 1])
            script_traj.append(['swapHands', 1])
            script_traj.append(['hotbar.1', 1])

            action = deepcopy(NOOP_ACTION)
            script_action = script_traj[self.collect_item_check_index]
            action[script_action[0]] = script_action[1]
            
            self.collect_item_check_index += 1
            if self.collect_item_check_index >= len(script_traj):
                self.collect_item_check_index = 0
                self.collect_item_check_flag = None
                return action, True
            return action, False

    def action_animal_hotbar(self, action, yolo_res):        
        # get maximize count animal
        animal_count = {}
        if len(yolo_res) > 0:
            for key in yolo_res:
                if yolo_res[key]['class'] not in animal_count:
                    animal_count[yolo_res[key]['class']] = 0
                animal_count[yolo_res[key]['class']] += 1
        
        max_animal = None
        if len(animal_count) > 0:
            animal_count_list = [ (count, animal) for animal, count in animal_count.items()]
            animal_count_list = sorted(animal_count_list)

            max_animal = animal_count_list[-1][1]
            max_animal_hotbar = self.animal_hotbar_dict[max_animal]

            action[max_animal_hotbar] = 1
        
        return action, max_animal

    def action_mask(self, action):
        action['attack'][0] = 0
        action['use'][0] = 0
        action['drop'][0] = 0
        action['inventory'][0] = 0
        action['ESC'] = 0
        action['sneak'] = 0

        for i in range(1, 10):
            action['hotbar.{}'.format(i)] = 0

        action['back'] = 0
        return action

    def action_mask_with_yolo(self, action, yolo_res):
        action = self.action_mask(action)

        # yolo intervene
        if len(yolo_res) > 0:
            self.mask_with_yolo_count = 1
            for key in yolo_res:
                if yolo_res[key]['size'][0] * yolo_res[key]['size'][1] < 10000:
                    self.record_mask_with_yolo_x = yolo_res[key]['pos'][0]
        
        if self.mask_with_yolo_count > 0:
            action['camera'] = [0, -int((self.MIDDLE_X - self.record_mask_with_yolo_x) / 12)]
            self.mask_with_yolo_count += 1
            if self.mask_with_yolo_count > self.max_mask_with_yolo_count:
                self.mask_with_yolo_count = 0
        
        # prevent look down
        if self.angle_x > LOOK_DOWN_ANGLE:
            action['camera'][0] = LOOK_DOWN_ANGLE - self.angle_x
        # prevent look up
        if self.angle_x < -LOOK_UP_ANGLE:
            action['camera'][0] = -LOOK_UP_ANGLE - self.angle_x
        return action

    def action_mask_for_attract_animal(self, action):
        if random.uniform(0, 1) < 0.95:
            action['sneak'] = 1
        if random.uniform(0, 1) < 0.1:
            action['jump'] = 1
        return action

    def action_mask_with_water(self, action, img):
        direction = self.water_direction(img)
        if direction != 0:
            action['camera'] = [0, direction]
            return action
        return action

    def action_mask_in_water(self, action, img):
        in_water = self.is_in_water(img)
        if in_water:
            action['jump'] = 1
            return action
        return action

    def action_mask_with_water_limit(self, action, img):
        direction = self.water_direction_limit(img)
        if direction != 0:
            action['camera'] = [0, direction]
            return action
        return action

    def pursue_action(self, action, yolo_res):
        # calcaulate animal count
        animal_count = {}
        if len(yolo_res) > 0:
            for key in yolo_res:
                if yolo_res[key]['class'] not in animal_count:
                    animal_count[yolo_res[key]['class']] = 0
                animal_count[yolo_res[key]['class']] += 1

        # get maximize count animal
        max_animal = None
        if len(animal_count) > 0:
            animal_count_list = [ (count, animal) for animal, count in animal_count.items()]
            animal_count_list = sorted(animal_count_list)
            max_animal = animal_count_list[-1][1]
            if max_animal == 'people' or max_animal == 'flower':
                max_animal = None

        # get maximize count animal pos
        animal_center_pos = []
        animal_window_size = []
        if max_animal:
            for key in yolo_res:
                if yolo_res[key]['class'] == max_animal:
                    animal_center_pos.append(yolo_res[key]['pos'])
                    animal_window_size.append(yolo_res[key]['size'][0] * yolo_res[key]['size'][1])

        # calcualaute camera action
        if len(animal_center_pos) > 0:
            self.pursue_count += 1
            animal_center_pos = np.array(animal_center_pos)
            animal_center_pos = np.mean(animal_center_pos, axis=0)
            animal_center_pos_x = animal_center_pos[0]
            self.record_animal_center_pos_x = animal_center_pos_x

            action = deepcopy(NOOP_ACTION)
            action['inventory'] = 0
            action['camera'] = [0 , -int((self.MIDDLE_X - animal_center_pos_x) / 8)]
            action["forward"] = 1
            action["jump"] = 1

            if max(animal_window_size) > MEAN_ANIMAL_WINDOW_SIZE[max_animal]:
                action['forward'] = 0
                action['jump'] = 0
                action['camera'] = [0 , 0]
                action, self.record_detect_animal = self.action_animal_hotbar(action, yolo_res)
                print('==============================')
                #self.save_video_record()
                return action, True
        
        # 累计跑
        if len(animal_center_pos) == 0 and self.pursue_count > 0:
            self.pursue_count += 1
            print('pursue_count: ', self.pursue_count)
            if self.pursue_count >= self.max_pursue_count:
                self.pursue_count = 0
            action['camera'] = [0 , -int((self.MIDDLE_X - self.record_animal_center_pos_x) / 8)]

        action, _ = self.action_animal_hotbar(action, yolo_res)
        return action, False

    def reset_agent_angle_action(self, img):
        action = deepcopy(NOOP_ACTION)
        if self.reset_agent_angle_action_flag == 0:
            self._get_target_y_angle()
             # angle check
            action['camera'] = [10, -self.angle_y + self.target_angle[1]]
            if abs(self.angle_x) == 90 and abs(self.target_angle[1] - self.angle_y) < 0.01:
                self.reset_agent_angle_action_flag = 1
        
        elif self.reset_agent_angle_action_flag == 1:
            action['attack'] = 1
            self.reset_agent_angle_action_flag = 2

        elif self.reset_agent_angle_action_flag == 2:
            # angle recheck
            turn_angle = self.get_angle(img)
            if abs(turn_angle) > 5:
                action['camera'] = [0, turn_angle]
                self.angle_y -= turn_angle
            self.reset_agent_angle_action_flag = 0
            return action, True

        return action, False
    
    def reset_agent_pos_angle_action(self, img, debug=False):
        ATTACK_FREQ = 8
        # 1. 先实现角度重置
        if not self.has_reset_angle:
            action, self.has_reset_angle = self.reset_agent_angle_action(img)
            if self.reset_agent_pos_angle_action_flag == ATTACK_FREQ:
                action['attack'] = 1
                self.reset_agent_pos_angle_action_flag = 0
            self.reset_agent_pos_angle_action_flag += 1

            self.center_x, self.center_y = None, None
        
        else: # 2. 坐标角度重置
            # 2.1. 获取agent坐标
            action = deepcopy(NOOP_ACTION)
            center_pos = self.get_center_pos(img)
            if debug:
                print('detected center_pos: ', center_pos)

            if center_pos is not None:
                center_x, center_y = center_pos
                
                # 2.2. 计算agent坐标与目标坐标的差值
                delta_x = self.MIDDLE_X - center_x
                delta_y = self.MIDDLE_Y - center_y

                # 2.3. 动作更新
                BIAS_X = 3 
                BIAS_Y = 7
                action['sneak'] = 1
                if delta_x > BIAS_X:
                    action['left'] = 1
                if delta_x < -BIAS_X:
                    action['right'] = 1
                if delta_y > BIAS_Y:
                    action['forward'] = 1
                if delta_y < -BIAS_Y:
                    action['back'] = 1
                
                if abs(delta_x) < BIAS_X and abs(delta_y) < BIAS_Y:
                    # 动物hotbar设置
                    action[self.animal_hotbar_dict[self.record_detect_animal]] = 1
                    self.has_reset_angle = False
                    self.reset_angle_count = 0
                    self.reset_agent_pos_angle_action_flag = 0
                    return action, True, self.reset_angle_count
            else: # 没有检测到的话，random walk
                action = deepcopy(NOOP_ACTION)
                action['sneak'] = 1
                if random.uniform(0, 1) > 0.5:
                    action['left'] = 1
                else:
                    action['right'] = 1
                if random.uniform(0, 1) > 0.5:
                    action['forward'] = 1
                else:
                    action['back'] = 1
                if self.reset_agent_pos_angle_action_flag == ATTACK_FREQ:
                    action['attack'] = 1
                self.reset_agent_pos_angle_action_flag = 0

        self.reset_angle_count += 1
        # 动物hotbar设置
        action[self.animal_hotbar_dict[self.record_detect_animal]] = 1

        if self.reset_angle_count > 40:
            self.reset_angle_count = 0
            self.has_reset_angle = False
            self.reset_agent_pos_angle_action_flag = 0
            return action, True, self.reset_angle_count
        return action, False, self.reset_angle_count

    def reset_agent_pos_angle_action_1(self, img, debug=False):
        ATTACK_FREQ = 8
        # 1. 先实现角度重置
        if not self.has_reset_angle:
            action, self.has_reset_angle = self.reset_agent_angle_action(img)
            if self.reset_agent_pos_angle_action_flag == ATTACK_FREQ:
                action['attack'] = 1
                self.reset_agent_pos_angle_action_flag = 0
            self.reset_agent_pos_angle_action_flag += 1

            self.center_x, self.center_y = None, None
        
        else: # 2. 坐标角度重置
            # 2.1. 获取agent坐标
            action = deepcopy(NOOP_ACTION)
            center_pos = self.get_center_pos(img)
            if debug:
                print('detected center_pos: ', center_pos)

            if center_pos is not None:
                center_x, center_y = center_pos
                
                # 2.2. 计算agent坐标与目标坐标的差值
                delta_x = self.MIDDLE_X - center_x

                # 2.3. 动作更新
                BIAS_X = 3 
                action['sneak'] = 1
                if delta_x > BIAS_X:
                    action['left'] = 1
                if delta_x < -BIAS_X:
                    action['right'] = 1

                if abs(delta_x) < BIAS_X:
                    # 动物hotbar设置
                    action[self.animal_hotbar_dict[self.record_detect_animal]] = 1
                    self.has_reset_angle = False
                    self.reset_angle_count = 0
                    self.reset_agent_pos_angle_action_flag = 0
                    return action, True, self.reset_angle_count
            else: # 没有检测到的话，random walk
                action = deepcopy(NOOP_ACTION)
                action['sneak'] = 1
                if random.uniform(0, 1) > 0.5:
                    action['left'] = 1
                else:
                    action['right'] = 1

                if self.reset_agent_pos_angle_action_flag == ATTACK_FREQ:
                    action['attack'] = 1
                self.reset_agent_pos_angle_action_flag = 0

        self.reset_angle_count += 1

        if self.reset_angle_count > 40:
            self.reset_angle_count = 0
            self.has_reset_angle = False
            self.reset_agent_pos_angle_action_flag = 0
            return action, True, self.reset_angle_count
        return action, False, self.reset_angle_count

    def reset_agent_pos_angle_action_v2(self, img, debug=False):
        N_DETECT_THRESHOLD = 5

        if self.reset_agent_pos_angle_action_v2_flag == 0:
            script_traj = []
            script_traj.append(['forward', 1])
            
            script_transition = script_traj[0]
            action = deepcopy(NOOP_ACTION)
            action[script_transition[0]] = script_transition[1]
            self.reset_agent_pos_angle_action_v2_flag = 1

        elif self.reset_agent_pos_angle_action_v2_flag < N_DETECT_THRESHOLD:
            action, done, _ = self.reset_agent_pos_angle_action(img)
            if done:
                if self.center_y is not None:  
                    if (self.MIDDLE_Y - self.center_y) < 3.:
                        self.reset_agent_pos_angle_action_v2_flag = 0
                        return action, True, None
                self.reset_agent_pos_angle_action_v2_flag += 1

        if self.reset_agent_pos_angle_action_v2_flag == N_DETECT_THRESHOLD:
            self.reset_agent_pos_angle_action_v2_flag = 0
            return action, True, None
        return action, False, None

    def reset_agent_pos_angle_action_v2_1(self, img, debug=False):
        # if self.pre_make_hole is None:
        #     self.pre_make_hole = self.is_snow_space(img)
        
        script_traj = []
        script_traj.append(['swapHands', 1])
        script_traj.append(['hotbar.6', 1])
        script_traj.append(['camera', [90 - self.angle_x, 0]])
        # if self.pre_make_hole: # 如果是雪地，预先打一下
        #     for _ in range(30):
        #         script_traj.append(['attack', 1])
        # for _ in range(30):
        #     script_traj.append(['attack', 1])
        # for _ in range(2):
        #     script_traj.append(['forward', 1])
        # for _ in range(7):
        #     for _ in range(2):
        #         script_traj.append(['left', 1,])
        #     for _ in range(4):
        #         script_traj.append(['right', 1])
        #     script_traj.append(['back', 1])
        #     for _ in range(2):
        #         script_traj.append(['left', 1])
        # for _ in range(5):
        #     script_traj.append(['back', 1, 'right', 1])
        # for _ in range(3):
        #     script_traj.append(['left', 1, 'sneak', 1])
        # for _ in range(2):
        #     script_traj.append(['forward', 1, 'sneak', 1])
        # for _ in range(10):
        #     script_traj.append(['jump', 1, 'use', 1])

        script_action = script_traj[self.plane_detect_fill_or_dig_v2_1_count]
        action = deepcopy(NOOP_ACTION)
        action[script_action[0]] = script_action[1]
        if len(script_action) == 4:
            action[script_action[2]] = script_action[3]

        self.plane_detect_fill_or_dig_v2_1_count += 1
        if self.plane_detect_fill_or_dig_v2_1_count >= len(script_traj):
            self.plane_detect_fill_or_dig_v2_1_count = 0
            self.pre_make_hole = None
            return action, True
        return action, False
    
    def reset_agent_pos_angle_action_v2_2(self, img, debug=False):
        if self.pre_make_hole is None:
            self.pre_make_hole = self.is_snow_space(img)
        
        script_traj = []
        script_traj.append(['hotbar.6', 1])
        script_traj.append(['camera', [90 - self.angle_x, 0]])
        if self.pre_make_hole: # 如果是雪地，预先打一下
            for _ in range(30):
                script_traj.append(['attack', 1])
        for _ in range(30):
            script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['forward', 1])
        for _ in range(7):
            for _ in range(2):
                script_traj.append(['left', 1,])
            for _ in range(4):
                script_traj.append(['right', 1])
            script_traj.append(['back', 1])
            for _ in range(2):
                script_traj.append(['left', 1])

        for _ in range(5):
            script_traj.append(['back', 1, 'left', 1])
        for _ in range(3):
            script_traj.append(['right', 1, 'sneak', 1])
        for _ in range(2):
            script_traj.append(['forward', 1, 'sneak', 1])
        for _ in range(10):
            script_traj.append(['jump', 1, 'use', 1])

        script_action = script_traj[self.plane_detect_fill_or_dig_v2_1_count]
        action = deepcopy(NOOP_ACTION)
        action[script_action[0]] = script_action[1]
        if len(script_action) == 4:
            action[script_action[2]] = script_action[3]

        self.plane_detect_fill_or_dig_v2_1_count += 1
        if self.plane_detect_fill_or_dig_v2_1_count >= len(script_traj):
            self.plane_detect_fill_or_dig_v2_1_count = 0
            self.pre_make_hole = None
            return action, True
        return action, False

    def plane_detect(self, img):
        DETECT_BLOCK = 5
        BLOCK_TAKEN_STEP = 5
        if self.plane_calibration_index == 0:
            self.plane_visual_direction = []
            self.plane_frame = []

        # define script action
        script_traj = []
        for _ in range(2):
            script_traj.append(['camera', [0, 0]])
        for _ in range(DETECT_BLOCK):
            for _ in range(BLOCK_TAKEN_STEP):
                script_traj.append(['right', 1])
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['camera', [0, 0]])

        # ======================================================
        # get taked action
        script_action = script_traj[self.plane_calibration_index]
        action = deepcopy(NOOP_ACTION)
        action[script_action[0]] = script_action[1]

        # optial flow
        _, diffs, ori_corner_poses = self.of.step(img)
        visual_direction = self.of.handle_direction_plane_detect(diffs, ori_corner_poses)
        if self.plane_calibration_init_count >= 3:
            self.plane_visual_direction.append(visual_direction)
            self.plane_frame.append(img.copy())
        
        # ======================================================
        # handle plane calibration
        self.plane_calibration_index += 1
        self.plane_calibration_init_count += 1
        if self.plane_calibration_index >= len(script_traj):
            # prehandle plane visual direction
            downside_block_plane = []
            upside_block_plane = []
            for i in range(DETECT_BLOCK):
                visual_directions = []
                for j in range(BLOCK_TAKEN_STEP+1):
                    visual_directions.append(self.plane_visual_direction[i*(BLOCK_TAKEN_STEP+1)+j])

                # handle downside 
                if Counter(visual_directions)['down'] > 2:
                    downside_block_plane.append(-1)
                else:
                    downside_block_plane.append(0)

                # handle upside
                if Counter(visual_directions)[None] > 2:
                    upside_block_plane.append(1)
                else:
                    upside_block_plane.append(0)
            
            frame_diffs = [0.]
            for idx in range(len(self.plane_frame)-1):
                frame_diffs.append(np.mean(np.abs(self.plane_frame[idx+1] - self.plane_frame[idx])>0))
            
            # rehandle plane visual direction
            block_frame_diffs = []
            for i in range(DETECT_BLOCK):
                block_frame_diff = []
                for j in range(BLOCK_TAKEN_STEP+1):
                    block_frame_diff.append(frame_diffs[i*(BLOCK_TAKEN_STEP+1)+j])
                
                if sum(block_frame_diff) / len(block_frame_diff) < 0.1:
                    block_frame_diffs.append(1)
                else:
                    block_frame_diffs.append(0)

            # final handle
            block_direction = []
            for upside, downside, frame_diff in zip(upside_block_plane, downside_block_plane, block_frame_diffs):
                if downside == -1:
                    block_direction.append(-1)
                elif upside == 1 or frame_diff == 1:
                    block_direction.append(1)
                else:
                    block_direction.append(0)

            for idx in range(len(block_direction)-1):
                block_direction[idx + 1] += block_direction[idx]

            self.plane_calibration_index = 0
            self.plane_calibration_init_count = 0
            return action, block_direction, True
        return action, None, False

    def plane_return(self, img):
        if self.last_img is None:
            self.last_img = img.copy()
            action = deepcopy(NOOP_ACTION)
            action['left'] = 1
            return action, False

        action = deepcopy(NOOP_ACTION)
        action['left'] = 1
        if np.mean(np.abs(img - self.last_img) > 0) < 0.1:
            action['jump'] = 1
        self.last_img = img.copy()

        self.plane_calibration_back_index += 1
        if self.plane_calibration_back_index >= self.plane_calibration_back_index_max:
            self.plane_calibration_back_index = 0
            return action, True
        return action, False

    def multi_plane_detect(self, img):
        STATUS_BUILD_FENCE = 0
        STATUS_PLANE_LEVEL_1 = 1
        STATUS_PLANE_LEVEL_1_RETURN = 2
        STATUS_PLANE_LEVEL_2_PRE = 3
        STATUS_PLANE_LEVEL_2 = 4
        STATUS_PLANE_LEVEL_2_RETURN = 5
        STATUS_PLANE_LEVEL_3_PRE = 6
        STATUS_PLANE_LEVEL_3 = 7
        STATUS_PLANE_LEVEL_3_RETURN = 8
        STATUS_PLANE_LEVEL_4_PRE = 9
        STATUS_PLANE_LEVEL_4 = 10
        STATUS_PLANE_LEVEL_4_RETURN = 11
        STATUS_PLANE_LEVEL_5_PRE = 12
        STATUS_PLANE_LEVEL_5 = 13
        STATUS_PLANE_LEVEL_5_RETURN = 14

        if self.multi_plane_detect_status == STATUS_BUILD_FENCE:
            script_traj = []
            script_traj.append(['camera', [-35, 0]])
            script_traj.append(['camera', [0, 90]])
            script_traj.append(['attack', 1])
            script_traj.append(['camera', [0, 0]]) # blank action
            script_traj.append(['camera', [0, 0]]) # blank action
            script_traj.append(['hotbar.1', 1])
            script_traj.append(['use', 1])
            script_traj.append(['camera', [0, 0]]) # blank action
            script_traj.append(['camera', [-self.angle_x, 90]]) # blank action

            script_action = script_traj[self.multi_plane_detect_index]
            action = deepcopy(NOOP_ACTION)
            action[script_action[0]] = script_action[1]

            self.multi_plane_detect_index += 1
            if self.multi_plane_detect_index == len(script_traj):
                self.multi_plane_detect_index = 0
                self.multi_plane_detect_status = STATUS_PLANE_LEVEL_1
        
        elif self.multi_plane_detect_status == STATUS_PLANE_LEVEL_1:
            action, block_direction, is_done = self.plane_detect(img)
            if is_done:
                self.multi_plane_detect_status = STATUS_PLANE_LEVEL_1_RETURN
                print(block_direction)
        
        elif self.multi_plane_detect_status == STATUS_PLANE_LEVEL_1_RETURN:
            action, is_done = self.plane_return(img)
            if is_done:
                self.multi_plane_detect_status = STATUS_PLANE_LEVEL_2_PRE
        
        elif self.multi_plane_detect_status == STATUS_PLANE_LEVEL_2_PRE:
            script_traj = []
            script_traj.append(['right', 1])
            script_traj.append(['right', 1])
            for _ in range(5):
                script_traj.append(['forward', 1])
            script_traj.append(['camera', [60, 0]])
            script_traj.append(['camera', [0, -90]])
            script_traj.append(['attack', 1])
            script_traj.append(['camera', [0, 0]]) # blank action
            script_traj.append(['camera', [0, 0]]) # blank action
            script_traj.append(['hotbar.1', 1])
            script_traj.append(['use', 1])
            script_traj.append(['camera', [0, 0]]) # blank action
            script_traj.append(['camera', [-self.angle_x, 90]]) # blank action

            script_action = script_traj[self.multi_plane_detect_index]
            action = deepcopy(NOOP_ACTION)
            action[script_action[0]] = script_action[1]

            self.multi_plane_detect_index += 1
            if self.multi_plane_detect_index == len(script_traj):
                self.multi_plane_detect_index = 0
                self.multi_plane_detect_status = STATUS_PLANE_LEVEL_2

        elif self.multi_plane_detect_status == STATUS_PLANE_LEVEL_2:
            action, block_direction, is_done = self.plane_detect(img)
            if is_done:
                self.multi_plane_detect_status = STATUS_PLANE_LEVEL_2_RETURN
                print(block_direction)

        elif self.multi_plane_detect_status == STATUS_PLANE_LEVEL_2_RETURN:
            action, is_done = self.plane_return(img)
            if is_done:
                self.multi_plane_detect_status = STATUS_PLANE_LEVEL_3_PRE

        elif self.multi_plane_detect_status == STATUS_PLANE_LEVEL_3_PRE:
            script_traj = []
            script_traj.append(['right', 1])
            script_traj.append(['right', 1])
            for _ in range(5):
                script_traj.append(['forward', 1])
            script_traj.append(['camera', [60, 0]])
            script_traj.append(['camera', [0, -90]])
            script_traj.append(['attack', 1])
            script_traj.append(['camera', [0, 0]]) # blank action
            script_traj.append(['camera', [0, 0]]) # blank action
            script_traj.append(['hotbar.1', 1])
            script_traj.append(['use', 1])
            script_traj.append(['camera', [0, 0]]) # blank action
            script_traj.append(['camera', [-self.angle_x, 90]]) # blank action

            script_action = script_traj[self.multi_plane_detect_index]
            action = deepcopy(NOOP_ACTION)
            action[script_action[0]] = script_action[1]

            self.multi_plane_detect_index += 1
            if self.multi_plane_detect_index == len(script_traj):
                self.multi_plane_detect_index = 0
                self.multi_plane_detect_status = STATUS_PLANE_LEVEL_3

        elif self.multi_plane_detect_status == STATUS_PLANE_LEVEL_3:
            action, block_direction, is_done = self.plane_detect(img)
            if is_done:
                self.multi_plane_detect_status = STATUS_PLANE_LEVEL_3_RETURN
                print(block_direction)

        elif self.multi_plane_detect_status == STATUS_PLANE_LEVEL_3_RETURN:
            action, is_done = self.plane_return(img)
            if is_done:
                self.multi_plane_detect_status = STATUS_PLANE_LEVEL_4_PRE
        
        elif self.multi_plane_detect_status == STATUS_PLANE_LEVEL_4_PRE:
            script_traj = []
            script_traj.append(['right', 1])
            script_traj.append(['right', 1])
            for _ in range(5):
                script_traj.append(['forward', 1])
            script_traj.append(['camera', [60, 0]])
            script_traj.append(['camera', [0, -90]])
            script_traj.append(['attack', 1])
            script_traj.append(['camera', [0, 0]]) # blank action
            script_traj.append(['camera', [0, 0]]) # blank action
            script_traj.append(['hotbar.1', 1])
            script_traj.append(['use', 1])
            script_traj.append(['camera', [0, 0]]) # blank action
            script_traj.append(['camera', [-self.angle_x, 90]]) # blank action

            script_action = script_traj[self.multi_plane_detect_index]
            action = deepcopy(NOOP_ACTION)
            action[script_action[0]] = script_action[1]

            self.multi_plane_detect_index += 1
            if self.multi_plane_detect_index == len(script_traj):
                self.multi_plane_detect_index = 0
                self.multi_plane_detect_status = STATUS_PLANE_LEVEL_4

        elif self.multi_plane_detect_status == STATUS_PLANE_LEVEL_4:
            action, block_direction, is_done = self.plane_detect(img)
            if is_done:
                self.multi_plane_detect_status = STATUS_PLANE_LEVEL_4_RETURN
                print(block_direction)

        elif self.multi_plane_detect_status == STATUS_PLANE_LEVEL_4_RETURN:
            action, is_done = self.plane_return(img)
            if is_done:
                self.multi_plane_detect_status = STATUS_PLANE_LEVEL_5_PRE
        
        elif self.multi_plane_detect_status == STATUS_PLANE_LEVEL_5_PRE:
            script_traj = []
            script_traj.append(['right', 1])
            script_traj.append(['right', 1])
            for _ in range(5):
                script_traj.append(['forward', 1])
            script_traj.append(['camera', [60, 0]])
            script_traj.append(['camera', [0, -90]])
            script_traj.append(['attack', 1])
            script_traj.append(['camera', [0, 0]]) # blank action
            script_traj.append(['camera', [0, 0]]) # blank action
            script_traj.append(['hotbar.1', 1])
            script_traj.append(['use', 1])
            script_traj.append(['camera', [0, 0]]) # blank action
            script_traj.append(['camera', [-self.angle_x, 90]]) # blank action

            script_action = script_traj[self.multi_plane_detect_index]
            action = deepcopy(NOOP_ACTION)
            action[script_action[0]] = script_action[1]

            self.multi_plane_detect_index += 1
            if self.multi_plane_detect_index == len(script_traj):
                self.multi_plane_detect_index = 0
                self.multi_plane_detect_status = STATUS_PLANE_LEVEL_5

        elif self.multi_plane_detect_status == STATUS_PLANE_LEVEL_5:
            action, block_direction, is_done = self.plane_detect(img)
            if is_done:
                self.multi_plane_detect_status = STATUS_PLANE_LEVEL_5_RETURN
                print(block_direction)

        elif self.multi_plane_detect_status == STATUS_PLANE_LEVEL_5_RETURN:
            action, is_done = self.plane_return(img)
            if is_done:
                self.multi_plane_detect_status = STATUS_BUILD_FENCE
                return action, True

        return action, False

    def plane_detect_via_fill(self, img):
        script_traj = []

        # 1.左转，抬头
        script_traj.append(['hotbar.6', 1])
        script_traj.append(['camera', [-10, -90]])
        # 2.走4个格
        for _ in range(4):
            for _ in range(5):
                script_traj.append(['forward', 1, 'use', 1])
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['camera', [0, 0]])
        # 3.右转
        script_traj.append(['camera', [0, 90]])
        script_traj.append(['forward', 1, 'use', 1])
        script_traj.append(['forward', 1, 'use', 1])
        script_traj.append(['camera', [0, 0]])
        # 4.右走4个格子
        for _ in range(4):
            for _ in range(5):
                script_traj.append(['right', 1, 'use', 1])
        # 5.向前走一个格子
        for _ in range(5):
                script_traj.append(['forward', 1, 'use', 1])
        # 6.左走4个格子
        for _ in range(4):
            for _ in range(5):
                script_traj.append(['left', 1, 'use', 1])
        # 7.向前走一个格子
        for _ in range(5):
                script_traj.append(['forward', 1, 'use', 1])
        # 8.右走4个格子
        for _ in range(4):
            for _ in range(5):
                script_traj.append(['right', 1, 'use', 1])
        # 9.向前走一个格子
        for _ in range(5):
                script_traj.append(['forward', 1, 'use', 1])
        # 10.左走4个格子
        for _ in range(4):
            for _ in range(5):
                script_traj.append(['left', 1, 'use', 1])
        # 11.回退2步
        script_traj.append(['back', 1])
        script_traj.append(['back', 1])
        
        action = deepcopy(NOOP_ACTION)
        script_action = script_traj[self.plane_detect_via_fill_index]
        if len(script_action) == 2:
            action[script_action[0]] = script_action[1]
        elif len(script_action) == 4:
            action[script_action[0]] = script_action[1]
            action[script_action[2]] = script_action[3]

        self.plane_detect_via_fill_index += 1
        if self.plane_detect_via_fill_index >= len(script_traj):
            self.plane_detect_via_fill_index = 0
            return action, True
        return action, False

    def plane_detect_via_fill_v2(self, img):
        N_ROW = 5
        N_COL = 5
        PLANE_DETECT_VIA_FILL_STATUS_DICT = {
            f'BLOCK_{j}_{i}': i+j*100 for i in range(N_ROW) for j in range(N_COL)
        }
        
        LEFT = -1
        RIGHT = 1
        script_traj = []
        action = deepcopy(NOOP_ACTION)
        if self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_0']:
            # 1.左转，抬头
            #script_traj.append(['swapHands', 1])
            script_traj.append(['hotbar.6', 1])
            script_traj.append(['camera', [-10, -90]])

            script_action = script_traj[self.plane_detect_via_fill_index]
            action[script_action[0]] = script_action[1]

            self.plane_detect_via_fill_index += 1
            if self.plane_detect_via_fill_index >= len(script_traj):
                self.plane_detect_via_fill_index = 0
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_1']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_2']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_2']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_3']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_3']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_4']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_4']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_4']
  
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_4']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_3']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_3']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_2']
            
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_2']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_1']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_0']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_0']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_0']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_1']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_1']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_2']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_2']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_3']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_3']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_4']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_4']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_4']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_4']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_3']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_3']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_2']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_2']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_1']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_0']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_0']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_0']:
            for _ in range(3):
                script_traj.append(['camera', [0, -30]])

            for _ in range(4):
                for _ in range(5):
                    script_traj.append(['forward', 1, 'use', 1])
                for _ in range(5):
                    script_traj.append(['back', 1])
                for _ in range(5):
                    script_traj.append(['left', 1])

            for _ in range(5):
                    script_traj.append(['forward', 1, 'use', 1])
            for _ in range(5):
                script_traj.append(['back', 1])

            script_action = script_traj[self.plane_detect_via_fill_index]
            action[script_action[0]] = script_action[1]
            if len(script_action) == 4:
                action[script_action[2]] = script_action[3]
            if len(script_action) == 6:
                action[script_action[2]] = script_action[3]
                action[script_action[4]] = script_action[5]

            self.plane_detect_via_fill_index += 1
            if self.plane_detect_via_fill_index >= len(script_traj):
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_0']
                self.plane_detect_via_fill_index = 0
                return action, True

        self.plane_detect_via_fill_last_img = img.copy()
        return action, False

    def plane_detect_via_fill_v2_1(self, img):
        N_ROW = 5
        N_COL = 5
        PLANE_DETECT_VIA_FILL_STATUS_DICT = {
            f'BLOCK_{j}_{i}': i+j*100 for i in range(N_ROW) for j in range(N_COL)
        }
        
        LEFT = -1
        RIGHT = 1
        script_traj = []
        action = deepcopy(NOOP_ACTION)
        if self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_0']:
            # 1.左转，抬头
            #script_traj.append(['swapHands', 1])
            script_traj.append(['hotbar.6', 1])
            script_traj.append(['camera', [-10, -90]])

            script_action = script_traj[self.plane_detect_via_fill_index]
            action[script_action[0]] = script_action[1]

            self.plane_detect_via_fill_index += 1
            if self.plane_detect_via_fill_index >= len(script_traj):
                self.plane_detect_via_fill_index = 0
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_1']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_2']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_2']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_3']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_3']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_4']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_4']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_4']
  
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_4']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_3']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_3']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_2']
            
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_2']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_1']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_0']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_0']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_0']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_1']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_1']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_2']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_2']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_3']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_3']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_4']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_4']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_4']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_4']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_3']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_3']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_2']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_2']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_1']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_0']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_0']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_0']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_1']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_2']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_2']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_3']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_3']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_4']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_4']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                return action, True

        self.plane_detect_via_fill_last_img = img.copy()
        return action, False

    def plane_detect_via_fill_v2_2(self, img):
        N_ROW = 5
        N_COL = 5
        PLANE_DETECT_VIA_FILL_STATUS_DICT = {
            f'BLOCK_{j}_{i}': i+j*100 for i in range(N_ROW) for j in range(N_COL)
        }
        
        LEFT = -1
        RIGHT = 1
        script_traj = []
        action = deepcopy(NOOP_ACTION)
        if self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_0']:
            # 1.左转，抬头
            #script_traj.append(['swapHands', 1])
            script_traj.append(['hotbar.6', 1])
            script_traj.append(['camera', [-10, -90]])

            script_action = script_traj[self.plane_detect_via_fill_index]
            action[script_action[0]] = script_action[1]

            self.plane_detect_via_fill_index += 1
            if self.plane_detect_via_fill_index >= len(script_traj):
                self.plane_detect_via_fill_index = 0
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_1']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_2']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_2']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_3']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_3']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_3']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_3']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_2']
            
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_2']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_1']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_0']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_0']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_0']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_1']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_1']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_2']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_2']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_3']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_3']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_3']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_3']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_2']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_2']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_1']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_0']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                return action, True
           
        self.plane_detect_via_fill_last_img = img.copy()
        return action, False

    def plane_detect_via_fill_v3(self, img):
        N_ROW = 5
        N_COL = 5
        PLANE_DETECT_VIA_FILL_STATUS_DICT = {
            f'BLOCK_{j}_{i}': i+j*100 for i in range(N_ROW) for j in range(N_COL)
        }

        LEFT = -1
        RIGHT = 1
        script_traj = []
        action = deepcopy(NOOP_ACTION)
        if self.plane_detect_via_fill_status == -1:
            script_traj.append(['hotbar.6', 1])
            script_traj.append(['camera', [-10, 0]])

            script_action = script_traj[self.plane_detect_via_fill_index]
            action[script_action[0]] = script_action[1]

            self.plane_detect_via_fill_index += 1
            if self.plane_detect_via_fill_index >= len(script_traj):
                self.plane_detect_via_fill_index = 0
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_2']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_2']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_1']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_0']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_0']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_0']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_0']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_0']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_0']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_1']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_2']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_2']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_3']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_3']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_4']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_4']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_4']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_4']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_4']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_4']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_4']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_4']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_4']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_4']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_3']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_3']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_2']
                return action, True

        self.plane_detect_via_fill_last_img = img.copy()
        return action, False

    def plane_detect_via_fill_v4(self, img):
        N_ROW = 5
        N_COL = 5
        PLANE_DETECT_VIA_FILL_STATUS_DICT = {
            f'BLOCK_{j}_{i}': i+j*100 for i in range(N_ROW) for j in range(N_COL)
        }

        LEFT = -1
        RIGHT = 1
        script_traj = []
        action = deepcopy(NOOP_ACTION)
        if self.plane_detect_via_fill_status == -1:
            script_traj.append(['hotbar.6', 1])
            script_traj.append(['camera', [-10, 0]])

            script_action = script_traj[self.plane_detect_via_fill_index]
            action[script_action[0]] = script_action[1]

            self.plane_detect_via_fill_index += 1
            if self.plane_detect_via_fill_index >= len(script_traj):
                self.plane_detect_via_fill_index = 0
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_4']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_4']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_4']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_4']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_4']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_4']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_4']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_4']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_3']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_3']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_2']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_2']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_1']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_0']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_0']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_0']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_0']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_0']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_0']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_1']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_2']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_2']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_3']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_3']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_4']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_4']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = -1
                return action, True

        self.plane_detect_via_fill_last_img = img.copy()
        return action, False

    def plane_detect_via_fill_v5_pre(self, img):
        N_ROW = 5
        N_COL = 5
        PLANE_DETECT_VIA_FILL_STATUS_DICT = {
            f'BLOCK_{j}_{i}': i+j*100 for i in range(N_ROW) for j in range(N_COL)
        }
        
        LEFT = -1
        RIGHT = 1
        script_traj = []
        action = deepcopy(NOOP_ACTION)
        if self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_0']:
            # 1.左转，抬头
            script_traj.append(['camera', [-10, -90]])

            script_action = script_traj[self.plane_detect_via_fill_index]
            action[script_action[0]] = script_action[1]

            self.plane_detect_via_fill_index += 1
            if self.plane_detect_via_fill_index >= len(script_traj):
                self.plane_detect_via_fill_index = 0
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_1']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_1']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_1']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_2']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_2']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_2']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_2']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_1']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_1']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_1']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_2']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_2']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = -1
                return action, True

        self.plane_detect_via_fill_last_img = img.copy()
        return action, False

    def plane_detect_via_fill_v5(self, img):
        N_ROW = 5
        N_COL = 5
        PLANE_DETECT_VIA_FILL_STATUS_DICT = {
            f'BLOCK_{j}_{i}': i+j*100 for i in range(N_ROW) for j in range(N_COL)
        }

        LEFT = -1
        RIGHT = 1
        script_traj = []
        action = deepcopy(NOOP_ACTION)
        if self.plane_detect_via_fill_status == -1:
            script_traj.append(['hotbar.6', 1])

            # clean
            script_traj.append(['camera', [-10, -90]])
            script_traj.append(['attack', 1])
            for _ in range(2):
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['camera', [-10, 0]])
            script_traj.append(['attack', 1])
            for _ in range(2):
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['camera', [-10, 0]])
            script_traj.append(['attack', 1])
            for _ in range(2):
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['camera', [-10, 0]])
            script_traj.append(['attack', 1])
            for _ in range(2):
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['camera', [-10, 0]])
            script_traj.append(['attack', 1])
            for _ in range(2):
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['camera', [-5, 25]])
            script_traj.append(['attack', 1])
            for _ in range(2):
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['camera', [13, 17]])
            script_traj.append(['attack', 1])
            for _ in range(2):
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['camera', [10, 43]])
            script_traj.append(['attack', 1])
            for _ in range(2):
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['camera', [10, 0]])
            script_traj.append(['attack', 1])
            for _ in range(2):
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['camera', [22, 5]])
            script_traj.append(['attack', 1])
            for _ in range(2):
                script_traj.append(['camera', [0, 0]])

            script_traj.append(['camera', [-10, 0]])
            script_action = script_traj[self.plane_detect_via_fill_index]
            action[script_action[0]] = script_action[1]

            self.plane_detect_via_fill_index += 1
            if self.plane_detect_via_fill_index >= len(script_traj):
                self.plane_detect_via_fill_index = 0
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_4']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_4']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_4']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_4']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_3']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_3']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_3']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_3']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_4']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_4']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_4']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_4']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_3']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_3']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_2']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_2']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_1']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_0']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_0']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_0']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_0']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_0']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_0']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_1']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_2']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_2']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_3']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_3']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_4']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_4_4']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = -1
                return action, True

        self.plane_detect_via_fill_last_img = img.copy()
        return action, False

    def plane_detect_via_fill_v5_pre_back(self, img):
        script_traj = []
        script_traj.append([self.animal_hotbar_dict[self.record_detect_animal], 1])
        script_traj.append(['swapHands', 1])

        script_action = script_traj[self.plane_detect_via_fill_v5_pre_index]
        action = deepcopy(NOOP_ACTION)
        action[script_action[0]] = script_action[1]

        self.plane_detect_via_fill_v5_pre_index += 1
        if self.plane_detect_via_fill_v5_pre_index == len(script_traj):
            self.plane_detect_via_fill_v5_pre_index = 0
            return action, True
        return action, False

    def plane_detect_via_fill_v5_pre_v2(self, img):
        N_ROW = 5
        N_COL = 5
        PLANE_DETECT_VIA_FILL_STATUS_DICT = {
            f'BLOCK_{j}_{i}': i+j*100 for i in range(N_ROW) for j in range(N_COL)
        }
        
        LEFT = -1
        RIGHT = 1
        script_traj = []
        action = deepcopy(NOOP_ACTION)
        if self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_0_0']:
            # 1.左转，抬头
            script_traj.append(['camera', [-10, 0]])

            script_action = script_traj[self.plane_detect_via_fill_index]
            action[script_action[0]] = script_action[1]

            self.plane_detect_via_fill_index += 1
            if self.plane_detect_via_fill_index >= len(script_traj):
                self.plane_detect_via_fill_index = 0
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_0']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_1']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_2']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_1_2']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_2']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_2']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_1']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_1']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, RIGHT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_0']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_2_0']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_0']
        
        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_0']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_1']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_1']:
            action, done = self._plane_detect_fill_or_dig_with_turn(img, LEFT)
            if done:
                self.plane_detect_via_fill_status = PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_2']

        elif self.plane_detect_via_fill_status == PLANE_DETECT_VIA_FILL_STATUS_DICT['BLOCK_3_2']:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_status = -1
                return action, True

        self.plane_detect_via_fill_last_img = img.copy()
        return action, False

    def _plane_detect_fill_or_dig(self, img):
        FRAME_DIFF_THREHOLD = 0.1
        UN_BLOCK_STATUS = 0
        BLOCK_STATUS = 1
        SUF_BLOCK_STATUS = 2
        PRE_BLOCK_STATUS = 3

        script_traj = []
        if self.plane_detect_via_fill_block_status == UN_BLOCK_STATUS:
            # cut forward tree
            for _ in range(4):
                script_traj.append(['camera', [-20, 0], 'attack', 1]) # FIXME: attack action
            for _ in range(7):
                script_traj.append(['attack', 1])
            for _ in range(4):
                script_traj.append(['camera', [20, 0], 'attack', 1]) # FIXME: attack action

            for _ in range(4):
                script_traj.append(['forward', 1, 'sneak', 1, 'attack', 1])
            for _ in range(9):
                script_traj.append(['forward', 1, 'use', 1, 'sneak', 1])
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['camera', [0, 0]])

            script_action = script_traj[self.plane_detect_via_fill_index]
            action = deepcopy(NOOP_ACTION)
            action[script_action[0]] = script_action[1]
            if len(script_action) == 4:
                action[script_action[2]] = script_action[3]
            if len(script_action) == 6:
                action[script_action[2]] = script_action[3]
                action[script_action[4]] = script_action[5]
        
        # elif self.plane_detect_via_fill_block_status == PRE_BLOCK_STATUS:
        #     action, done, _ = self.reset_agent_pos_angle_action_1(img)
        #     if done:
        #         action['camera'] = [-10, 0]
        #         self.plane_detect_via_fill_index = 0
        #         self.plane_detect_via_fill_block_status = BLOCK_STATUS

        elif self.plane_detect_via_fill_block_status == BLOCK_STATUS:
            script_traj.append(['camera', [-30, 0]])
            n_attack = 25
            if self.plane_detect_fill_or_dig_count >= 2:
                n_attack = 80
            if self.plane_detect_fill_or_dig_count >= 3:
                n_attack = 160
            
            if self.plane_detect_fill_or_dig_count < 5:
                for _ in range(n_attack):
                    script_traj.append(['attack', 1])
                script_traj.append(['camera', [30, 0]])
                for _ in range(5):
                    script_traj.append(['camera', [0, 0]])

            elif self.plane_detect_fill_or_dig_count == 5: # 向下敲，坐标复位
                script_traj.append(['camera', [30, -90]])
                script_traj.append(['use', 1])
                script_traj.append(['camera', [10, 90]])
                for _ in range(30):
                    script_traj.append(['attack', 1])
                for _ in range(2):
                    script_traj.append(['forward', 1])

                for _ in range(5):
                    for _ in range(3):
                        script_traj.append(['left', 1,])
                    for _ in range(6):
                        script_traj.append(['right', 1])
                    script_traj.append(['back', 1])
                    for _ in range(3):
                        script_traj.append(['left', 1])
                for _ in range(2):
                    for _ in range(2):
                        script_traj.append(['left', 1,])
                    for _ in range(4):
                        script_traj.append(['right', 1])
                    script_traj.append(['back', 1])
                    for _ in range(2):
                        script_traj.append(['left', 1])
                # TODO: 挖坑坐标系复位
                for _ in range(3):
                    script_traj.append(['back', 1])
                for _ in range(2):
                    script_traj.append(['forward', 1, 'sneak', 1])
                for _ in range(10):
                    script_traj.append(['jump', 1, 'use', 1])
                script_traj.append(['camera', [-10, 0]])

            else:
                if self.plane_detect_fill_or_dig_count <= 7:
                    n_attack = 25
                elif self.plane_detect_fill_or_dig_count <= 9:
                    n_attack = 80
                else:
                    n_attack = 160
                
                for _ in range(n_attack):
                    script_traj.append(['attack', 1])
                script_traj.append(['camera', [30, 0]])
                for _ in range(5):
                    script_traj.append(['camera', [0, 0]])
            
            script_action = script_traj[self.plane_detect_via_fill_index]
            action = deepcopy(NOOP_ACTION)
            action[script_action[0]] = script_action[1]
            if len(script_action) == 4:
                action[script_action[2]] = script_action[3]

        elif self.plane_detect_via_fill_block_status == SUF_BLOCK_STATUS:
            for _ in range(3):
                script_traj.append(['forward', 1])
            for _ in range(1):
                script_traj.append(['forward', 1, 'sneak', 1])
            script_traj.append(['camera', [0, 0]])

            script_action = script_traj[self.plane_detect_via_fill_index]
            action = deepcopy(NOOP_ACTION)
            action[script_action[0]] = script_action[1]
            if len(script_action) == 4:
                action[script_action[2]] = script_action[3]

        self.plane_detect_via_fill_index += 1
        if self.plane_detect_via_fill_index >= len(script_traj):
            self.plane_detect_via_fill_index = 0

            # 被卡住检测
            if self.plane_detect_via_fill_block_status == UN_BLOCK_STATUS:
                is_block = False
                frame_diff = np.mean( np.abs(self.plane_detect_via_fill_last_img -img) > 0 )
                if frame_diff < FRAME_DIFF_THREHOLD:
                    is_block = True

                if is_block:
                    self.plane_detect_via_fill_block_status = BLOCK_STATUS
                else:
                    self.plane_detect_fill_or_dig_count = 0
                    return action, True
            
            elif self.plane_detect_via_fill_block_status == BLOCK_STATUS:
                self.plane_detect_via_fill_block_status = SUF_BLOCK_STATUS

            elif self.plane_detect_via_fill_block_status == SUF_BLOCK_STATUS:
                is_block = False
                frame_diff = np.mean( np.abs(self.plane_detect_via_fill_last_img -img) > 0 )
                if frame_diff < FRAME_DIFF_THREHOLD:
                    is_block = True

                if is_block:
                    self.plane_detect_fill_or_dig_count += 1
                    self.plane_detect_via_fill_block_status = BLOCK_STATUS
                else:
                    self.plane_detect_via_fill_block_status = UN_BLOCK_STATUS
                    self.plane_detect_fill_or_dig_count = 0
                    return action, True

        return action, False

    def _plane_detect_fill_or_dig_with_turn(self, img, turn):
        script_traj = []
        if not self.plane_detect_via_fill_block_turn_status:
            for _ in range(3):
                script_traj.append(['camera', [0, 30*turn]])
            
            script_action = script_traj[self.plane_detect_via_fill_index]
            action = deepcopy(NOOP_ACTION)
            action[script_action[0]] = script_action[1]

            self.plane_detect_via_fill_index += 1
            if self.plane_detect_via_fill_index >= len(script_traj):
                self.plane_detect_via_fill_index = 0
                self.plane_detect_via_fill_block_turn_status = True
        else:
            action, done = self._plane_detect_fill_or_dig(img)
            if done:
                self.plane_detect_via_fill_block_turn_status = False
                return action, True

        return action, False

    def make_hole(self):
        script_traj = []
        # 切hotbar
        script_traj.append(['swapHands', 1])
        # 打第一层
        script_traj.append(['camera', [90 - self.angle_x, 0]])
        for _ in range(30):
            script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['forward', 1])
        script_traj.append(['camera', [-20, -90]])
        for _ in range(30):
            script_traj.append(['attack', 1])
        for _ in range(10):
            script_traj.append(['forward', 1])
        script_traj.append(['camera', [0, -90]])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(10):
            script_traj.append(['forward', 1])
        for _ in range(30):
            script_traj.append(['attack', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        script_traj.append(['camera', [0, -90]])
        for _ in range(10):
            script_traj.append(['forward', 1])
        for _ in range(30):
            script_traj.append(['attack', 1])
        # 打第二层
        script_traj.append(['camera', [90 - self.angle_x, 0]])
        for _ in range(30):
            script_traj.append(['attack', 1])
        for _ in range(10):
            script_traj.append(['left', 1, 'back', 1])
        script_traj.append(['camera', [-20, 0]])
        for _ in range(10):
            script_traj.append(['forward', 1])
        for _ in range(30):
            script_traj.append(['attack', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        script_traj.append(['camera', [0, -90]])
        for _ in range(10):
            script_traj.append(['forward', 1])
        for _ in range(30):
            script_traj.append(['attack', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        script_traj.append(['camera', [0, -90]])
        for _ in range(10):
            script_traj.append(['forward', 1])
        for _ in range(30):
            script_traj.append(['attack', 1])
        # 敲边出去&封路
        script_traj.append(['camera', [-70, 90]])
        for _ in range(3):
            script_traj.append(['forward', 1])
        for _ in range(3):
            script_traj.append(['right', 1])
        for _ in range(30):
            script_traj.append(['attack', 1])
        for _ in range(8):
            script_traj.append(['forward', 1, 'jump', 1])
        script_traj.append(['camera', [90 - self.angle_x, 0], 'hotbar.6', 1])
        for _ in range(8):
            script_traj.append(['jump', 1, 'use', 1])

        script_action = script_traj[self.make_hole_index]
        action = deepcopy(NOOP_ACTION)
        action[script_action[0]] = script_action[1]
        if len(script_action) == 4:
            action[script_action[2]] = script_action[3]

        self.make_hole_index += 1
        if self.make_hole_index >= len(script_traj):
            self.make_hole_index = 0
            return action, True
        return action, False

    def make_hole_v2(self, img):
        """
        plane_detect_via_fill_v4
        """
        if self.pre_make_hole is None:
            self.pre_make_hole = self.is_snow_space(img)

        script_traj = []
        # 切hotbar, 向前走一格
        script_traj.append(['swapHands', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        # 打第一块 (右下角)
        script_traj.append(['camera', [90 - self.angle_x, 0]])
        if self.pre_make_hole: # 如果是雪地，预先打一下
            for _ in range(30):
                script_traj.append(['attack', 1])
        for _ in range(30):
            script_traj.append(['attack', 1])
        for _ in range(3):
            script_traj.append(['back', 1])
        for _ in range(10):
            script_traj.append(['forward', 1])
        script_traj.append(['camera', [0, 0]])
        for _ in range(30):
            script_traj.append(['attack', 1])
        # 打第二块（左下角）
        script_traj.append(['camera', [0, -90]])
        for _ in range(3):
            script_traj.append(['forward', 1])
        script_traj.append(['camera', [-50, 0]])
        for _ in range(30):
            script_traj.append(['attack', 1, 'forward', 1])
        script_traj.append(['camera', [30, 0]])
        for _ in range(30):
            script_traj.append(['attack', 1, 'forward', 1])
        # 打第三块（左中角）
        for _ in range(5):
            script_traj.append(['forward', 1])
        script_traj.append(['camera', [-30, 90]])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(30):
            script_traj.append(['attack', 1, 'forward', 1])
        script_traj.append(['camera', [30, 0]])
        for _ in range(30):
            script_traj.append(['attack', 1, 'forward', 1])
        # 打第四块（右中角）
        for _ in range(5):
            script_traj.append(['forward', 1])
        script_traj.append(['camera', [-30, 90]])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(30):
            script_traj.append(['attack', 1, 'forward', 1])
        script_traj.append(['camera', [30, 0]])
        for _ in range(30):
            script_traj.append(['attack', 1, 'forward', 1])
        # 打第5块（右上角）
        for _ in range(5):
            script_traj.append(['forward', 1])
        script_traj.append(['camera', [-30, -90]])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(30):
            script_traj.append(['attack', 1, 'forward', 1])
        script_traj.append(['camera', [30, 0]])
        for _ in range(30):
            script_traj.append(['attack', 1, 'forward', 1])
        # 打第6块（左上角）
        for _ in range(5):
            script_traj.append(['forward', 1])
        script_traj.append(['camera', [0, -90]])
        for _ in range(10):
            script_traj.append(['forward', 1])
        for _ in range(30):
            script_traj.append(['attack', 1, 'forward', 1])
        script_traj.append(['camera', [-30, 0]])
        for _ in range(30):
            script_traj.append(['attack', 1, 'forward', 1])
        script_traj.append(['camera', [30, 0]])
        # 敲边出去&封路
        for _ in range(10):
            script_traj.append(['forward', 1])
        script_traj.append(['camera', [-140, 0]])
        for _ in range(3):
            script_traj.append(['right', 1])
        for _ in range(30):
            script_traj.append(['attack', 1])
        script_traj.append(['camera', [70, 0]])
        for _ in range(30):
            script_traj.append(['attack', 1])
        for _ in range(8):
            script_traj.append(['forward', 1, 'jump', 1])
        script_traj.append(['camera', [90 - self.angle_x, 0], 'hotbar.6', 1])
        for _ in range(10):
            script_traj.append(['jump', 1, 'use', 1])

        script_action = script_traj[self.make_hole_index]
        action = deepcopy(NOOP_ACTION)
        action[script_action[0]] = script_action[1]
        if len(script_action) == 4:
            action[script_action[2]] = script_action[3]

        self.make_hole_index += 1
        if self.make_hole_index >= len(script_traj):
            self.make_hole_index = 0
            self.pre_make_hole = None
            return action, True
        return action, False

    def make_hole_v3(self, img):
        if self.is_stone_for_make_hole is None:
            self.is_stone_for_make_hole = self.is_stone_space(img)
        if self.is_stone_for_make_hole:
            action, done = self._make_hole_v3_stone(img)
        else:
            action, done = self._make_hole_v3_other(img)
        
        if done:
            self.is_stone_for_make_hole = None
        return action, done

    def _make_hole_v3_other(self, img):
        """
        for plane_detect_via_fill_v5
        """
        N_ATTACK = 23
        if self.pre_make_hole is None:
            self.pre_make_hole = self.is_snow_space(img)

        script_traj = []
        # 打第一块 (左上角)
        script_traj.append(['camera', [90 - self.angle_x, 30], 'hotbar.6', 1])
        script_traj.append(['camera', [0, 30]])
        script_traj.append(['camera', [0, 30]])
        if self.pre_make_hole: # 如果是雪地，预先打一下
            for _ in range(30):
                script_traj.append(['attack', 1])

        script_traj.append(['camera', [0, 0]])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1])
        for _ in range(3):
            script_traj.append(['forward', 1])
        for _ in range(7):
            for _ in range(2):
                script_traj.append(['left', 1,])
            for _ in range(4):
                script_traj.append(['right', 1])
            script_traj.append(['back', 1])
            for _ in range(2):
                script_traj.append(['left', 1])
        for _ in range(3):
            script_traj.append(['right', 1])

        script_traj.append(['camera', [0, 0]])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1])
        # 第一块补块
        script_traj.append(['camera', [-30, 0]]) # -30
        script_traj.append(['use', 1])
        script_traj.append(['camera', [-50, 0]]) # -80
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        script_traj.append(['camera', [0, -30]]) 
        script_traj.append(['camera', [0, -30]]) 
        script_traj.append(['camera', [0, -30]]) 
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        script_traj.append(['camera', [50, 0]]) # -30
        script_traj.append(['use', 1])
        # 打第二块，右上角
        script_traj.append(['camera', [-20, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        for _ in range(10):
            script_traj.append(['forward', 1, 'right', 1])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1])
        script_traj.append(['camera', [30, 0]]) # -20
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1])
        # 第二块补块
        for _ in range(5):
            script_traj.append(['forward', 1, 'camera', [-2, 0]]) # -30
        script_traj.append(['use', 1])
        script_traj.append(['camera', [-50, 0]]) # -80
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        script_traj.append(['camera', [0, -30]])
        script_traj.append(['camera', [0, -30]])
        script_traj.append(['camera', [0, -30]])
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        script_traj.append(['camera', [50, 0]]) # -30
        script_traj.append(['use', 1])
        # 打第三块，右中角
        script_traj.append(['camera', [10, 30]]) # -20
        script_traj.append(['camera', [0, 30]]) # -20
        script_traj.append(['camera', [0, 30]]) # -20
        script_traj.append(['camera', [0, 30]]) # -20
        script_traj.append(['camera', [0, 30]]) # -20
        script_traj.append(['camera', [0, 30]]) # -20
        for _ in range(10):
            script_traj.append(['forward', 1, 'left', 1])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1, 'forward', 1])
        script_traj.append(['camera', [-30, 0]]) # -50
        for _ in range(N_ATTACK-10):
            script_traj.append(['attack', 1, 'forward', 1])
        for _ in range(10):
            script_traj.append(['attack', 1])
        # 第三块补块
        for _ in range(5): # wait
            script_traj.append(['camera', [0, 0]])
        for _ in range(10):
            script_traj.append(['forward', 1, 'right', 1])
        script_traj.append(['camera', [-30, -30]])
        script_traj.append(['camera', [0, -30]])
        script_traj.append(['camera', [0, -30]])
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 打第四块，左中角
        script_traj.append(['camera', [10, 30]]) # -20
        script_traj.append(['camera', [10, 30]]) # -20
        script_traj.append(['camera', [10, 30]]) # -20
        script_traj.append(['camera', [10, 30]]) # -20
        script_traj.append(['camera', [10, 30]]) # -20
        script_traj.append(['camera', [10, 30]]) # -20
        for _ in range(10):
            script_traj.append(['forward', 1, 'left', 1])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1, 'forward', 1])
        script_traj.append(['camera', [-30, 0]]) # -50
        for _ in range(N_ATTACK-10):
            script_traj.append(['attack', 1, 'forward', 1])
        for _ in range(10):
            script_traj.append(['attack', 1])
        # 第四块补块
        for _ in range(10): # wait
            script_traj.append(['camera', [0, 0]])
        for _ in range(3):
            script_traj.append(['forward', 1, 'left', 1])
        script_traj.append(['camera', [-30, 0]]) # -80
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1, 'left', 1])
        # 打第五块，左下角
        script_traj.append(['camera', [20, -30]]) # -20
        script_traj.append(['camera', [20, -30]]) # -20
        script_traj.append(['camera', [20, -30]]) # -20
        for _ in range(10):
            script_traj.append(['forward', 1, 'right', 1])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1, 'forward', 1, 'right', 1])
        script_traj.append(['camera', [-30, 0]]) # -50
        for _ in range(N_ATTACK-10):
            script_traj.append(['attack', 1, 'forward', 1, 'right', 1])
        for _ in range(10):
            script_traj.append(['attack', 1])
        # # 第五块补块
        for _ in range(10): # wait
            script_traj.append(['camera', [0, 0]])
        for _ in range(3):
            script_traj.append(['forward', 1, 'left', 1])
        script_traj.append(['camera', [-30, 0]]) # -80
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1, 'left', 1])
        script_traj.append(['camera', [0, 30]]) # -80
        script_traj.append(['camera', [0, 30]]) # -80
        script_traj.append(['camera', [0, 30]]) # -80
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 打第六块，右下角
        script_traj.append(['camera', [10, 30]]) # -20
        script_traj.append(['camera', [10, 30]]) # -20
        script_traj.append(['camera', [10, 30]]) # -20
        script_traj.append(['camera', [10, 30]]) # -20
        script_traj.append(['camera', [10, 30]]) # -20
        script_traj.append(['camera', [10, 30]]) # -20
        for _ in range(10):
            script_traj.append(['forward', 1, 'right', 1])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1, 'forward', 1, 'right', 1])
        script_traj.append(['camera', [-30, 0]]) # -50
        for _ in range(N_ATTACK-10):
            script_traj.append(['attack', 1, 'forward', 1, 'right', 1])
        for _ in range(10):
            script_traj.append(['attack', 1])
        # 第六块补块
        for _ in range(10): # wait
            script_traj.append(['camera', [0, 0]])
        for _ in range(5):
            script_traj.append(['forward', 1])
        script_traj.append(['camera', [-30, 90]]) # -80
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 走一会，让动物掉下来
        for _ in range(5):
            script_traj.append(['forward', 1, 'right', 1])
        script_traj.append(['camera', [-10, 30]]) # -90
        script_traj.append(['camera', [0, 30]]) # -90
        script_traj.append(['camera', [0, 30]]) # -90
        for _ in range(15):
            script_traj.append(['forward', 1, 'left', 1])
        script_traj.append(['camera', [0, 90]])
        script_traj.append(['hotbar.9', 1])
        script_traj.append(['swapHands', 1])
        script_traj.append(['hotbar.6', 1])
        for _ in range(35):
            script_traj.append(['forward', 1, 'left', 1])
        # 敲边出去&封路
        script_traj.append(['camera', [-90, -90]])
        for _ in range(3):
            script_traj.append(['right', 1])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1, 'forward', 1])
        script_traj.append(['camera', [10, 0]])
        for _ in range(3):
            script_traj.append(['right', 1])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1, 'forward', 1])
        script_traj.append(['camera', [10, 0]])
        for _ in range(3):
            script_traj.append(['right', 1])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1, 'forward', 1])
        script_traj.append(['camera', [70, 0]])
        for _ in range(N_ATTACK+10):
            script_traj.append(['attack', 1])
        for _ in range(6):
            script_traj.append(['forward', 1, 'jump', 1])
        script_traj.append(['camera', [90 - self.angle_x, 0], 'forward', 1, 'sneak', 1])
        script_traj.append(['forward', 1, 'sneak', 1])
        for _ in range(10):
            script_traj.append(['jump', 1, 'use', 1])

        script_action = script_traj[self.make_hole_index]
        action = deepcopy(NOOP_ACTION)
        action[script_action[0]] = script_action[1]
        if len(script_action) == 4:
            action[script_action[2]] = script_action[3]
        if len(script_action) == 6:
            action[script_action[2]] = script_action[3]
            action[script_action[4]] = script_action[5]

        self.make_hole_index += 1
        if self.make_hole_index >= len(script_traj):
            self.make_hole_index = 0
            self.pre_make_hole = None
            return action, True
        return action, False

    def _make_hole_v3_stone(self, img):
        """
        for plane_detect_via_fill_v5
        """
        N_ATTACK = 23
        if self.pre_make_hole is None:
            self.pre_make_hole = self.is_snow_space(img)

        script_traj = []
        # 打第一块 (左上角)
        script_traj.append(['camera', [90 - self.angle_x, 90], 'hotbar.6', 1])
        if self.pre_make_hole: # 如果是雪地，预先打一下
            for _ in range(30):
                script_traj.append(['attack', 1])

        script_traj.append(['camera', [0, 0]])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1])
        for _ in range(3):
            script_traj.append(['forward', 1])
        for _ in range(7):
            for _ in range(2):
                script_traj.append(['left', 1,])
            for _ in range(4):
                script_traj.append(['right', 1])
            script_traj.append(['back', 1])
            for _ in range(2):
                script_traj.append(['left', 1])
        for _ in range(3):
            script_traj.append(['right', 1])

        script_traj.append(['camera', [0, 0]])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1])
        # 第一块补块
        script_traj.append(['camera', [-30, 0]]) # -30
        script_traj.append(['use', 1])
        script_traj.append(['camera', [-50, 0]]) # -80
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        script_traj.append(['camera', [0, -30]]) 
        script_traj.append(['camera', [0, -30]]) 
        script_traj.append(['camera', [0, -30]])  
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        script_traj.append(['camera', [50, 0]]) # -30
        script_traj.append(['use', 1])
        # 打第二块，右上角
        script_traj.append(['camera', [-20, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        for _ in range(10):
            script_traj.append(['forward', 1, 'right', 1])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1])
        script_traj.append(['camera', [30, 0]]) # -20
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1])
        # 第二块补块
        for _ in range(5):
            script_traj.append(['forward', 1, 'camera', [-2, 0]]) # -30
        script_traj.append(['use', 1])
        script_traj.append(['camera', [-50, 0]]) # -80
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        script_traj.append(['camera', [0, -30]])
        script_traj.append(['camera', [0, -30]])
        script_traj.append(['camera', [0, -30]])
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        script_traj.append(['camera', [50, 0]]) # -30
        script_traj.append(['use', 1])
        # 打第三块，右中角
        script_traj.append(['camera', [-20, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        for _ in range(10):
            script_traj.append(['forward', 1, 'left', 1])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1, 'forward', 1])
        script_traj.append(['camera', [30, 0]]) # -20
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1, 'forward', 1])
        # 第三块补块
        for _ in range(10):
            script_traj.append(['forward', 1, 'right', 1])
        script_traj.append(['camera', [-30, -30]]) # -80
        script_traj.append(['camera', [-30, -30]]) # -80
        script_traj.append(['camera', [0, -30]]) # -80
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 打第四块，左中角
        script_traj.append(['camera', [30, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        for _ in range(10):
            script_traj.append(['forward', 1, 'left', 1])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1, 'forward', 1])
        script_traj.append(['camera', [30, 0]]) # -20
        for _ in range(N_ATTACK-10):
            script_traj.append(['attack', 1, 'forward', 1])
        for _ in range(10):
            script_traj.append(['attack', 1])
        # 第四块补块
        for _ in range(3):
            script_traj.append(['forward', 1, 'left', 1])
        script_traj.append(['camera', [-60, 0]]) # -80
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1, 'left', 1])
        # 打第五块，左下角
        script_traj.append(['camera', [30, -30]]) # -50
        script_traj.append(['camera', [0, -30]]) # -50
        script_traj.append(['camera', [0, -30]]) # -50
        for _ in range(5):
            script_traj.append(['forward', 1, 'right', 1])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1, 'forward', 1, 'right', 1])
        script_traj.append(['camera', [30, 0]]) # -20
        for _ in range(N_ATTACK-10):
            script_traj.append(['attack', 1, 'forward', 1, 'right', 1])
        for _ in range(10):
            script_traj.append(['attack', 1])
        # 第五块补块
        for _ in range(3):
            script_traj.append(['forward', 1, 'left', 1])
        script_traj.append(['camera', [-60, 0]]) # -80
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1, 'left', 1])
        script_traj.append(['camera', [0, 30]]) # -80
        script_traj.append(['camera', [0, 30]]) # -80
        script_traj.append(['camera', [0, 30]]) # -80
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 打第六块，右下角
        script_traj.append(['camera', [30, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        script_traj.append(['camera', [0, 30]]) # -50
        for _ in range(10):
            script_traj.append(['forward', 1, 'right', 1])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1, 'forward', 1, 'right', 1])
        script_traj.append(['camera', [30, 0]]) # -20
        for _ in range(N_ATTACK-10):
            script_traj.append(['attack', 1, 'forward', 1, 'right', 1])
        for _ in range(10):
            script_traj.append(['attack', 1])
        # 第六块补块
        for _ in range(5):
            script_traj.append(['forward', 1])
        script_traj.append(['camera', [-60, 90]]) # -80
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 走一会，让动物掉下来
        for _ in range(5):
            script_traj.append(['forward', 1, 'right', 1])
        script_traj.append(['camera', [-10, 90]]) # -90
        for _ in range(15):
            script_traj.append(['forward', 1, 'left', 1])
        script_traj.append(['camera', [0, 90]])
        script_traj.append(['hotbar.9', 1])
        script_traj.append(['swapHands', 1])
        script_traj.append(['hotbar.6', 1])
        for _ in range(35):
            script_traj.append(['forward', 1, 'left', 1])
        # 敲边出去&封路
        script_traj.append(['camera', [-90, -90]])
        for _ in range(3):
            script_traj.append(['right', 1])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1, 'forward', 1])
        script_traj.append(['camera', [10, 0]])
        for _ in range(3):
            script_traj.append(['right', 1])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1, 'forward', 1])
        script_traj.append(['camera', [10, 0]])
        for _ in range(3):
            script_traj.append(['right', 1])
        for _ in range(N_ATTACK):
            script_traj.append(['attack', 1, 'forward', 1])
        script_traj.append(['camera', [70, 0]])
        for _ in range(N_ATTACK+2):
            script_traj.append(['attack', 1])
        for _ in range(6):
            script_traj.append(['forward', 1, 'jump', 1])
        script_traj.append(['camera', [90 - self.angle_x, 0], 'forward', 1, 'sneak', 1])
        script_traj.append(['forward', 1, 'sneak', 1])
        for _ in range(10):
            script_traj.append(['jump', 1, 'use', 1])

        script_action = script_traj[self.make_hole_index]
        action = deepcopy(NOOP_ACTION)
        action[script_action[0]] = script_action[1]
        if len(script_action) == 4:
            action[script_action[2]] = script_action[3]
        if len(script_action) == 6:
            action[script_action[2]] = script_action[3]
            action[script_action[4]] = script_action[5]

        self.make_hole_index += 1
        if self.make_hole_index >= len(script_traj):
            self.make_hole_index = 0
            self.pre_make_hole = None
            return action, True
        return action, False

    def stuck_action(self):
        script_traj = []
        script_traj.append(['hotbar.6', 1, 'back', 1])
        script_traj.append(['camera', [90 - self.angle_x, 0], 'back', 1])
        for _ in range(10):
            script_traj.append(['jump', 1, 'use', 1])
        for _ in range(10):
            script_traj.append(['forward', 1])
        script_traj.append(['hotbar.1', 1, 'camera', [-90, 0]])

        script_action = script_traj[self.stuck_action_index]
        action = deepcopy(NOOP_ACTION)
        action[script_action[0]] = script_action[1]
        if len(script_action) == 4:
            action[script_action[2]] = script_action[3]

        self.stuck_action_index += 1
        if self.stuck_action_index >= len(script_traj):
            self.stuck_action_index = 0
            return action, True
        return action, False

    def minimum_guarantee(self):
        self._get_target_y_angle()

        script_traj = []
        script_traj.append(['hotbar.1', 1, 'camera', [65 - self.angle_x, self.target_angle[1] - self.angle_y]])
        for _ in range(4):
            for _ in range(3):
                for _ in range(5):
                    script_traj.append(['right', 1])
                script_traj.append(['attack', 1])
                for _ in range(2):  # blank action
                    script_traj.append(['camera', [0, 0]])
                script_traj.append(['use', 1])
            script_traj.append(['camera', [0, 90]])
        
        script_traj.append(['forward', 1, 'left', 1])
        for _ in range(4):
            for _ in range(3):
                for _ in range(5):
                    script_traj.append(['right', 1])
                script_traj.append(['attack', 1])
                for _ in range(2):  # blank action
                    script_traj.append(['camera', [0, 0]])
                script_traj.append(['use', 1])
            script_traj.append(['camera', [0, 90]])
        
        script_traj.append(['forward', 1, 'left', 1])
        for _ in range(4):
            for _ in range(3):
                for _ in range(5):
                    script_traj.append(['right', 1])
                script_traj.append(['attack', 1])
                for _ in range(2):  # blank action
                    script_traj.append(['camera', [0, 0]])
                script_traj.append(['use', 1])
            script_traj.append(['camera', [0, 90]])

        
        script_action = script_traj[self.minimum_guarantee_index]
        action = deepcopy(NOOP_ACTION)
        action[script_action[0]] = script_action[1]
        if len(script_action) == 4:
            action[script_action[2]] = script_action[3]

        self.minimum_guarantee_index += 1
        if self.minimum_guarantee_index >= len(script_traj):
            self.minimum_guarantee_index = 0
            return action, True
        return action, False

    # ===================
    #  micro action
    # ===================
    def wait(self):
        action = deepcopy(NOOP_ACTION)
        if self.wait_count >= self.max_wait_count - 5:
            action['back'] = 1
        if self.wait_count == self.max_wait_count - 1:
            action['camera'] = [0, 180]
        self.wait_count += 1

        # 动物hotbar设置
        action[self.animal_hotbar_dict[self.record_detect_animal]] = 1
        return action, self.wait_count >= self.max_wait_count

    def load_back_wait_animal(self, yolo_res):
        script_traj = []
        script_traj.append(['camera', [0, 30]])
        script_traj.append(['camera', [0, 30]])
        script_traj.append(['camera', [0, 25]])
        script_traj.append(['camera', [0, 15]])
        script_traj.append(['camera', [0, 10]])
        script_traj.append(['camera', [0, 10]])
        for _ in range(12):
            script_traj.append(['camera', [0, 10]])
        for _ in range(12):
            script_traj.append(['camera', [0, -10]])

        script_action = script_traj[self.load_back_wait_animal_index]
        action = deepcopy(NOOP_ACTION)
        action[script_action[0]] = script_action[1]
        self.load_back_wait_animal_index += 1

        if self.record_detect_animal:
            action[self.animal_hotbar_dict[self.record_detect_animal]] = 1

        animal_count = {}
        if len(yolo_res) > 0 and self.load_back_wait_animal_index > 6:
            for key in yolo_res:
                if yolo_res[key]['class'] not in animal_count:
                    animal_count[yolo_res[key]['class']] = 0
                animal_count[yolo_res[key]['class']] += 1

        # get maximize count animal
        max_animal = None
        if len(animal_count) > 0:
            animal_count_list = [ (count, animal) for animal, count in animal_count.items()]
            animal_count_list = sorted(animal_count_list)
            max_animal = animal_count_list[-1][1]
            if max_animal != self.record_detect_animal:
                max_animal = None

        # get maximize count animal pos
        animal_center_pos = []
        animal_window_size = []
        if max_animal:
            for key in yolo_res:
                if yolo_res[key]['class'] == max_animal:
                    animal_center_pos.append(yolo_res[key]['pos'])
                    animal_window_size.append(yolo_res[key]['size'][0] * yolo_res[key]['size'][1])

        # calcualaute camera action
        if len(animal_center_pos) > 0:
            self.pursue_count += 1
            animal_center_pos = np.array(animal_center_pos)
            animal_center_pos = np.mean(animal_center_pos, axis=0)
            animal_center_pos_x = animal_center_pos[0]
            self.record_animal_center_pos_x = animal_center_pos_x

            action = deepcopy(NOOP_ACTION)
            action['camera'] = [0 , -int((self.MIDDLE_X - animal_center_pos_x) / 8)]

            if max(animal_window_size) > MEAN_ANIMAL_WINDOW_SIZE[max_animal] // 10:
                action['camera'] = [0, 180]
                self.load_back_wait_animal_index = 0
                return action, True

        if self.load_back_wait_animal_index >= len(script_traj):
            self.load_back_wait_animal_index = 0
            action['camera'] = [0, 180]
            return action, True
        return action, False

    # ===================
    #  Follow Human Pen
    # ===================
    def pen_script(self):
        action = self.pen_traj[self.pen_action_index]
        self.pen_action_index += 1
        if self.pen_action_index == len(self.pen_traj):
            self.pen_action_index = 0
            return action, True, len(self.pen_traj), len(self.pen_traj)
        return action, False, self.pen_action_index, len(self.pen_traj)

    def pen_script_rabbit(self):
        action = self.pen_traj_rabbit[self.pen_action_index]
        self.pen_action_index += 1
        if self.pen_action_index == len(self.pen_traj_rabbit):
            self.pen_action_index = 0
            return action, True, len(self.pen_traj_rabbit), len(self.pen_traj_rabbit)
        return action, False, self.pen_action_index, len(self.pen_traj_rabbit)

    def get_pen_traj(self):
        script_traj = []
        # 1.
        script_traj.append(['camera', [-45, 0]])
        # 2.
        script_traj.append(['use', 1]) 
        # 3.
        for _ in range(15):
            script_traj.append(['forward', 1])
        # 4.
        for _ in range(4):
            script_traj.append(['back', 1])
        # 5.
        for _ in range(5):
            for _ in range(4):
                script_traj.append(['right', 1])
            script_traj.append(['use', 1])
        # 6.
        for _ in range(5):
            for _ in range(4):
                script_traj.append(['back', 1])
            script_traj.append(['use', 1])
        # 7.
        for _ in range(20):
            script_traj.append(['left', 1])
        # 8.
        for _ in range(20):
            script_traj.append(['forward', 1])
        # 9.
        for _ in range(5):
            for _ in range(4):
                script_traj.append(['back', 1])
            script_traj.append(['use', 1])
        # 10.
        for _ in range(2):
            for _ in range(4):
                script_traj.append(['right', 1])
            script_traj.append(['use', 1])
        # 11.
        script_traj.append(['hotbar.2', 1])
        for _ in range(4):
            script_traj.append(['right', 1])
        script_traj.append(['use', 1])
        # 12.
        script_traj.append(['hotbar.1', 1])
        for _ in range(2):
            for _ in range(4):
                script_traj.append(['right', 1])
            script_traj.append(['use', 1])
        # 13.
        for _ in range(4):
            script_traj.append(['back', 1])

        script_actions = []
        for script_transition in script_traj:
            action = deepcopy(NOOP_ACTION)
            action[script_transition[0]] = script_transition[1]
            script_actions.append(action)

        return script_actions

    def get_pen_traj_v2(self):
        script_traj = []
        # 1. 将食物切换到左手
        script_traj.append(['swapHands', 1])
        script_traj.append(['hotbar.1', 1])
        # 2. 向后转
        for _ in range(6):
            script_traj.append(['camera', [0, 30]])
        # ----上侧----
        # 3. 建造前方篱笆，垂直坐标对齐
        script_traj.append(['camera', [-35, 0]])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1]) 
        script_traj.append(['camera', [-10, 0]])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1]) 
        script_traj.append(['camera', [-10, 0]])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['camera', [10, 0]])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1]) 
        # ----
        for _ in range(25):
            script_traj.append(['forward', 1])
        # 4. 后退
        for _ in range(4):
            script_traj.append(['back', 1])
        # 5. 想右侧建造篱笆
        for _ in range(3):
            for _ in range(5):
                script_traj.append(['right', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):
                script_traj.append(['camera', [0, 0]]) # blank action
            script_traj.append(['use', 1])
        # ----右侧----
        # 6. 右转建个篱笆
        script_traj.append(['camera', [15, 90]])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # 7. 左走，建篱笆
        for _ in range(5):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # 8. 向前走，卡位
        for _ in range(5):
            script_traj.append(['forward', 1])
        # 9. 后退, 右走
        for _ in range(4):
            script_traj.append(['back', 1])
        for _ in range(2):
            script_traj.append(['right', 1]) 
        # 10. 右走，建篱笆
        for _ in range(3):
            for _ in range(5):
                script_traj.append(['right', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):
                script_traj.append(['camera', [0, 0]]) # blank action
            script_traj.append(['use', 1])
        # ----下侧----
        # 11. 右转建个篱笆
        script_traj.append(['camera', [0, 90]])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # 12. 左走，建篱笆
        for _ in range(5):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # 13. 向前走，卡位
        for _ in range(5):
            script_traj.append(['forward', 1])
        # 14. 后退, 右走
        for _ in range(4):
            script_traj.append(['back', 1])
        for _ in range(2):
            script_traj.append(['right', 1]) 
        # 15. 右走，建篱笆
        for _ in range(3):
            for _ in range(5):
                script_traj.append(['right', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):
                script_traj.append(['camera', [0, 0]]) # blank action
            script_traj.append(['use', 1])
        # ----左侧----
        # 16. 右转建个篱笆
        script_traj.append(['camera', [0, 90]])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # 17. 左走，建篱笆
        for _ in range(5):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # 18. 向前走，卡位
        for _ in range(5):
            script_traj.append(['forward', 1])
        # 19. 后退, 右走
        for _ in range(4):
            script_traj.append(['back', 1])
        for _ in range(2):
            script_traj.append(['right', 1]) 
        # 20. 右走，建篱笆
        for _ in range(5):
            script_traj.append(['right', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # 21. 建门
        script_traj.append(['hotbar.2', 1])
        for _ in range(5):
            script_traj.append(['right', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # 22. 建篱笆
        script_traj.append(['hotbar.1', 1])
        for _ in range(5):
            script_traj.append(['right', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # 23. 建最后篱笆
        for _ in range(5):
            script_traj.append(['right', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # -----出门----
        # 24. 走到门前，开门
        for _ in range(8):
            script_traj.append(['left', 1])
        for _ in range(2):
            script_traj.append(['forward', 1])
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        # 25. 向前走，出门
        for _ in range(6):
            script_traj.append(['forward', 1])
        # 26. 回头关门
        for _ in range(6):
            script_traj.append(['camera', [0, 30]])
        script_traj.append(['use', 1])
        # 27. 抬头
        script_traj.append(['camera', [-30, 0]])
        # -1.
        script_traj.append(['camera', [0, 0]])

        script_actions = []
        for script_transition in script_traj:
            action = deepcopy(NOOP_ACTION)
            action[script_transition[0]] = script_transition[1]
            script_actions.append(action)

        return script_actions

    def get_pen_traj_v3(self):
        script_traj = []
        # 0、 向前走一小段距离
        for _ in range(6):
            script_traj.append(['forward', 1])
        # 1. 将食物切换到左手
        script_traj.append(['swapHands', 1])
        script_traj.append(['hotbar.1', 1])
        # 2. 向左转
        for _ in range(3):
            script_traj.append(['camera', [0, -30]])
        script_traj.append(['camera', [-35, 0]])
        # 3. 向前走2格
        for _ in range(2):
            for _ in range(5):
                script_traj.append(['forward', 1])
        # 4. 建篱笆
        script_traj.append(['attack', 1])
        for _ in range(2): # blank action
            script_traj.append(['camera', [0, 0]]) 
        script_traj.append(['use', 1]) 
        # 5. 向右侧建篱笆（2个）
        for _ in range(2):
            for _ in range(5):
                script_traj.append(['right', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 6. 右转 建个篱笆
        for _ in range(3):
            script_traj.append(['camera', [0, 30]])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 7. 左侧走，建卡角篱笆
        for _ in range(4):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 8. 向前走，卡位
        for _ in range(5):
            script_traj.append(['forward', 1])
        # 9. 后退, 右走
        for _ in range(3):
            script_traj.append(['back', 1])
        for _ in range(2):
            script_traj.append(['right', 1]) 
        # 10. 向右侧建篱笆（3个）
        for _ in range(3):
            for _ in range(5):
                script_traj.append(['right', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 11. 右转 建个篱笆
        for _ in range(3):
            script_traj.append(['camera', [0, 30]])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 12. 左侧走，建卡角篱笆
        for _ in range(4):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 13. 向前走，卡位
        for _ in range(5):
            script_traj.append(['forward', 1])
        # 14. 后退, 右走
        for _ in range(3):
            script_traj.append(['back', 1])
        for _ in range(2):
            script_traj.append(['right', 1])
        # 15. 向右侧建篱笆（2个）
        for _ in range(2):
            for _ in range(5):
                script_traj.append(['right', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 16. 右转 建个篱笆
        for _ in range(3):
            script_traj.append(['camera', [0, 30]])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 17. 左侧走，建卡角篱笆
        for _ in range(4):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 18. 向前走，卡位
        for _ in range(5):
            script_traj.append(['forward', 1])
        # 19. 后退, 右走
        for _ in range(3):
            script_traj.append(['back', 1])
        for _ in range(2):
            script_traj.append(['right', 1]) 
        # 20. 右走，建篱笆
        for _ in range(5):
            script_traj.append(['right', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # 21. 建门
        script_traj.append(['hotbar.2', 1])
        for _ in range(5):
            script_traj.append(['right', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # 22. 建篱笆
        script_traj.append(['hotbar.1', 1])
        for _ in range(5):
            script_traj.append(['right', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # 23. 建最后篱笆
        for _ in range(5):
            script_traj.append(['right', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # -----出门----
        # 24. 走到门前，开门
        for _ in range(8):
            script_traj.append(['left', 1])
        for _ in range(2):
            script_traj.append(['forward', 1])
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        # 25. 向前走，出门
        for _ in range(6):
            script_traj.append(['forward', 1])
        # 26. 回头关门
        for _ in range(6):
            script_traj.append(['camera', [0, 30]])
        script_traj.append(['use', 1])
        # 27. 后退、抬头
        for _ in range(4):
            script_traj.append(['back', 1])
        script_traj.append(['camera', [-30, 0]])
        # -1.
        for _ in range(5):
            script_traj.append(['camera', [0, 0]])

        script_actions = []
        for script_transition in script_traj:
            action = deepcopy(NOOP_ACTION)
            action[script_transition[0]] = script_transition[1]
            script_actions.append(action)

        return script_actions

    def get_pen_traj_v4(self):
        script_traj = []
        # 1. 将食物切换到左手
        script_traj.append(['hotbar.1', 1])
        # script_traj.append(['swapHands', 1])
        # script_traj.append(['hotbar.1', 1]) 
        # 2.初始位置标定
        script_traj.append(['camera', [-25, 0]])
        for _ in range(5):
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['forward', 1, 'sneak', 1, 'use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(2):
            script_traj.append(['back', 1])
        # 3. 向右走3个格子，建一层篱笆
        for _ in range(3):
            for _ in range(5):
                script_traj.append(['right', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 4. 向左走2个格子（牵引动物）, 向右走两个格子
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['left', 1])
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['right', 1])
        # 5. 右转，建一个标定篱笆
        for _ in range(3):
            script_traj.append(['camera', [0, 30]])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 6. 卡位，建角篱笆
        for _ in range(4):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(3):
            script_traj.append(['back', 1])
        for _ in range(2):
            script_traj.append(['right', 1]) 
        # 7. 向右走2个格子，建一层篱笆
        for _ in range(2):
            for _ in range(5):
                script_traj.append(['right', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 8. 向左走2个格子（牵引动物）,向右走2个格子
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['left', 1])
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['right', 1])
        # 9. 右转，建一个标定篱笆
        for _ in range(3):
            script_traj.append(['camera', [0, 30]])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 10. 卡位，建角篱笆
        for _ in range(4):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(3):
            script_traj.append(['back', 1])
        for _ in range(2):
            script_traj.append(['right', 1]) 
        # 11. 向右走2个格子，建一层篱笆
        for _ in range(2):
            for _ in range(5):
                script_traj.append(['right', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 12. 向左走2个格子（牵引动物）,向右走2个格子
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['left', 1])
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['right', 1])
        # 13. 右转，建一个标定篱笆
        for _ in range(3):
            script_traj.append(['camera', [0, 30]])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 14. 卡位，建角篱笆
        for _ in range(4):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(3):
            script_traj.append(['back', 1])
        for _ in range(2):
            script_traj.append(['right', 1]) 
        # 15. 切hotbar，建门
        script_traj.append(['hotbar.2', 1])
        for _ in range(5):
            script_traj.append(['right', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # 16. 向右建1个篱笆
        script_traj.append(['hotbar.1', 1])
        for _ in range(5):
            script_traj.append(['right', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # 17. 走到门前，开门
        for _ in range(5):
            script_traj.append(['left', 1])
        for _ in range(2):
            script_traj.append(['forward', 1])
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        # 18. 向前走，出门
        for _ in range(6):
            script_traj.append(['forward', 1])
        # 19. 回头关门
        for _ in range(6):
            script_traj.append(['camera', [0, 30]])
        script_traj.append(['use', 1])
        # 20. 后退、抬头
        for _ in range(4):
            script_traj.append(['back', 1])
        script_traj.append(['camera', [-30, 0]])
        # -1.
        for _ in range(3):
            script_traj.append(['camera', [0, 0]])

        script_actions = []
        for script_transition in script_traj:
            action = deepcopy(NOOP_ACTION)
            action[script_transition[0]] = script_transition[1]
            if len(script_transition) == 4:
                action[script_transition[2]] = script_transition[3]
            if len(script_transition) == 6:
                action[script_transition[2]] = script_transition[3]
                action[script_transition[4]] = script_transition[5]
            script_actions.append(action)

        return script_actions

    def get_pen_traj_v4_1(self):
        script_traj = []
        # 1. 将食物切换到左手
        script_traj.append(['hotbar.1', 1])
        # script_traj.append(['swapHands', 1])
        # script_traj.append(['hotbar.1', 1]) 
        # 2.初始位置标定
        script_traj.append(['camera', [-20, 90]])
        for _ in range(5):
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['back', 1, 'sneak', 1,'use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(2):
            script_traj.append(['back', 1])
        # 3. 向右走3个格子，建一层篱笆
        for _ in range(3):
            for _ in range(5):
                script_traj.append(['right', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 4. 向左走2个格子（牵引动物）, 向右走两个格子
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['left', 1])
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['right', 1])
        # 5. 右转，建一个标定篱笆
        for _ in range(3):
            script_traj.append(['camera', [0, 30]])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 6. 卡位，建角篱笆
        for _ in range(4):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(3):
            script_traj.append(['back', 1])
        for _ in range(2):
            script_traj.append(['right', 1]) 
        # 7. 向右走2个格子，建一层篱笆
        for _ in range(2):
            for _ in range(5):
                script_traj.append(['right', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 8. 向左走2个格子（牵引动物）,向右走2个格子
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['left', 1])
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['right', 1])
        # 9. 右转，建一个标定篱笆
        for _ in range(3):
            script_traj.append(['camera', [0, 30]])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 10. 卡位，建角篱笆
        for _ in range(4):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(3):
            script_traj.append(['back', 1])
        for _ in range(2):
            script_traj.append(['right', 1]) 
        # 11. 向右走2个格子，建一层篱笆
        for _ in range(2):
            for _ in range(5):
                script_traj.append(['right', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 12. 向左走2个格子（牵引动物）,向右走2个格子
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['left', 1])
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['right', 1])
        # 13. 右转，建一个标定篱笆
        for _ in range(3):
            script_traj.append(['camera', [0, 30]])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 14. 卡位，建角篱笆
        for _ in range(4):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(3):
            script_traj.append(['back', 1])
        for _ in range(2):
            script_traj.append(['right', 1]) 
        # 15. 切hotbar，建门
        script_traj.append(['hotbar.2', 1])
        for _ in range(5):
            script_traj.append(['right', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # 16. 向右建1个篱笆
        script_traj.append(['hotbar.1', 1])
        for _ in range(5):
            script_traj.append(['right', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # 17. 走到门前，开门
        for _ in range(5):
            script_traj.append(['left', 1])
        for _ in range(2):
            script_traj.append(['forward', 1])
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        # 18. 向前走，出门
        for _ in range(6):
            script_traj.append(['forward', 1])
        # 19. 回头关门
        for _ in range(6):
            script_traj.append(['camera', [0, 30]])
        script_traj.append(['use', 1])
        # 20. 后退、抬头
        for _ in range(4):
            script_traj.append(['back', 1])
        script_traj.append(['camera', [-30, 0]])
        # -1.
        for _ in range(3):
            script_traj.append(['camera', [0, 0]])

        script_actions = []
        for script_transition in script_traj:
            action = deepcopy(NOOP_ACTION)
            action[script_transition[0]] = script_transition[1]
            if len(script_transition) == 4:
                action[script_transition[2]] = script_transition[3]
            if len(script_transition) == 6:
                action[script_transition[2]] = script_transition[3]
                action[script_transition[4]] = script_transition[5]
            script_actions.append(action)

        return script_actions

    def get_pen_traj_v4_2(self):
        script_traj = []
        # 1. 将食物切换到左手
        script_traj.append(['hotbar.1', 1])
        # script_traj.append(['swapHands', 1])
        # script_traj.append(['hotbar.1', 1]) 
        # 2.初始位置标定
        script_traj.append(['camera', [-20, 0]])
        for _ in range(5):
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['back', 1, 'sneak', 1,'use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(2):
            script_traj.append(['back', 1])
        # 3. 向左走3个格子，建一层篱笆
        for _ in range(3):
            for _ in range(5):
                script_traj.append(['left', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 4. 向左走2个格子（牵引动物）, 向右走两个格子
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['left', 1])
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['right', 1])
        # 5. 左转，建一个标定篱笆
        for _ in range(3):
            script_traj.append(['camera', [0, -30]])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 6. 卡位，建角篱笆
        for _ in range(4):
            script_traj.append(['right', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(3):
            script_traj.append(['back', 1])
        for _ in range(2):
            script_traj.append(['left', 1]) 
        # 7. 向左走2个格子，建一层篱笆
        for _ in range(2):
            for _ in range(5):
                script_traj.append(['left', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 8. 向左走2个格子（牵引动物）,向右走2个格子
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['left', 1])
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['right', 1])
        # 9. 左转，建一个标定篱笆
        for _ in range(3):
            script_traj.append(['camera', [0, -30]])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 10. 卡位，建角篱笆
        for _ in range(4):
            script_traj.append(['right', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(3):
            script_traj.append(['back', 1])
        for _ in range(2):
            script_traj.append(['left', 1]) 
        # 11. 向左走2个格子，建一层篱笆
        for _ in range(2):
            for _ in range(5):
                script_traj.append(['left', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 12. 向左走2个格子（牵引动物）,向右走2个格子
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['left', 1])
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['right', 1])
        # 13. 左转，建一个标定篱笆
        for _ in range(3):
            script_traj.append(['camera', [0, -30]])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 14. 卡位，建角篱笆
        for _ in range(4):
            script_traj.append(['right', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(3):
            script_traj.append(['back', 1])
        for _ in range(2):
            script_traj.append(['left', 1]) 
        # 15. 切hotbar，建门
        script_traj.append(['hotbar.2', 1])
        for _ in range(5):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # 16. 向左建1个篱笆
        script_traj.append(['hotbar.1', 1])
        for _ in range(5):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # 17. 走到门前，开门
        for _ in range(5):
            script_traj.append(['right', 1])
        for _ in range(2):
            script_traj.append(['forward', 1])
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        # 18. 向前走，出门
        for _ in range(6):
            script_traj.append(['forward', 1])
        # 19. 回头关门
        for _ in range(6):
            script_traj.append(['camera', [0, 30]])
        script_traj.append(['use', 1])
        # 20. 后退、抬头
        for _ in range(4):
            script_traj.append(['back', 1])
        script_traj.append(['camera', [-30, 0]])
        # -1.
        for _ in range(3):
            script_traj.append(['camera', [0, 0]])

        script_actions = []
        for script_transition in script_traj:
            action = deepcopy(NOOP_ACTION)
            action[script_transition[0]] = script_transition[1]
            if len(script_transition) == 4:
                action[script_transition[2]] = script_transition[3]
            if len(script_transition) == 6:
                action[script_transition[2]] = script_transition[3]
                action[script_transition[4]] = script_transition[5]
            script_actions.append(action)

        return script_actions

    def get_pen_traj_v4_3(self):
        script_traj = []
        # 1. 将食物切换到左手
        script_traj.append(['hotbar.1', 1])
        # script_traj.append(['swapHands', 1])
        # script_traj.append(['hotbar.1', 1]) 
        # 2.初始位置标定
        script_traj.append(['camera', [-20, 0]])
        for _ in range(3):
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['back', 1, 'sneak', 1])
        for _ in range(2):
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['back', 1, 'sneak', 1,'use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(2):
            script_traj.append(['back', 1])
        # 3. 向右走2个格子，建一层篱笆
        for _ in range(2):
            for _ in range(5):
                script_traj.append(['right', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 4. 向左走2个格子（牵引动物）, 向右走两个格子
        # 5. 右转，建一个标定篱笆
        for _ in range(3):
            script_traj.append(['camera', [0, 30]])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 6. 卡位，建角篱笆
        for _ in range(4):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(3):
            script_traj.append(['back', 1])
        for _ in range(2):
            script_traj.append(['right', 1]) 
        # 7. 向右走1个格子，建一层篱笆
        for _ in range(1):
            for _ in range(5):
                script_traj.append(['right', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 8. 向左走2个格子（牵引动物）,向右走2个格子
        # 9. 右转，建一个标定篱笆
        for _ in range(3):
            script_traj.append(['camera', [0, 30]])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 10. 卡位，建角篱笆
        for _ in range(4):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(3):
            script_traj.append(['back', 1])
        for _ in range(2):
            script_traj.append(['right', 1]) 
        # 11. 向右走1个格子，建一层篱笆
        for _ in range(1):
            for _ in range(5):
                script_traj.append(['right', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 12. 向左走2个格子（牵引动物）,向右走2个格子
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['left', 1])
        # for _ in range(2):
        #     for _ in range(5):
        #         script_traj.append(['right', 1])
        # 13. 右转，建一个标定篱笆
        for _ in range(3):
            script_traj.append(['camera', [0, 30]])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        # 14. 卡位，建角篱笆
        for _ in range(4):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(3):
            script_traj.append(['back', 1])
        for _ in range(2):
            script_traj.append(['right', 1]) 
        # 15. 切hotbar，建门
        script_traj.append(['hotbar.2', 1])
        for _ in range(5):
            script_traj.append(['right', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        script_traj.append(['use', 1])
        # 16. 向右建0个篱笆
        script_traj.append(['hotbar.1', 1])
        # 17. 走到门前，开门
        for _ in range(2):
            script_traj.append(['forward', 1])
        script_traj.append(['use', 1])
        for _ in range(2):
            script_traj.append(['camera', [0, 0]]) # blank action
        # 18. 向前走，出门
        for _ in range(6):
            script_traj.append(['forward', 1])
        # 19. 回头关门
        for _ in range(6):
            script_traj.append(['camera', [0, 30]])
        script_traj.append(['use', 1])
        # 20. 后退、抬头
        for _ in range(4):
            script_traj.append(['back', 1])
        script_traj.append(['camera', [-30, 0]])
        # -1.
        for _ in range(3):
            script_traj.append(['camera', [0, 0]])

        script_actions = []
        for script_transition in script_traj:
            action = deepcopy(NOOP_ACTION)
            action[script_transition[0]] = script_transition[1]
            if len(script_transition) == 4:
                action[script_transition[2]] = script_transition[3]
            if len(script_transition) == 6:
                action[script_transition[2]] = script_transition[3]
                action[script_transition[4]] = script_transition[5]
            script_actions.append(action)

        return script_actions

    def get_pen_traj_v5(self):
        script_traj = []
        # 1. 将食物切换到左手
        script_traj.append(['hotbar.1', 1])
        # script_traj.append(['swapHands', 1])
        # script_traj.append(['hotbar.1', 1]) 
        # 2.初始位置标定
        script_traj.append(['camera', [-25, 90]])
        for _ in range(5):
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['forward', 1, 'sneak', 1, 'use', 1])
        for _ in range(5):
            script_traj.append(['forward', 1])
        for _ in range(2):
            script_traj.append(['back', 1])
        script_traj.append(['camera', [0, -90]])
        # 3. 先建个篱笆，后退1格，建篱笆
        script_traj.append(['use', 1])
        for _ in range(5):
            script_traj.append(['back', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        for _ in range(4):
            script_traj.append(['forward', 1])
        for _ in range(2):
            script_traj.append(['back', 1])
        # 4. 左转后退，建篱笆
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
        # 5. 左转后退，建篱笆
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
        # 6. 左转后退，建篱笆
        script_traj.append(['camera', [0, -90]])
        for idx in range(5):
            script_traj.append(['back', 1])
            if idx >= 2:
                script_traj.append(['attack', 1])
                for _ in range(2):  # blank action
                    script_traj.append(['camera', [0, 0]])
                script_traj.append(['use', 1])
        for _ in range(5):
            script_traj.append(['right', 1])
        script_traj.append(['camera', [0, -90]])
        script_traj.append(['hotbar.2', 1])
        for _ in range(5):
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        script_traj.append(['hotbar.1', 1])
        for _ in range(2):
            for _ in range(5):
                script_traj.append(['left', 1])
            script_traj.append(['attack', 1])
            for _ in range(2):  # blank action
                script_traj.append(['camera', [0, 0]])
            script_traj.append(['use', 1])
        # 7. 封口
        script_traj.append(['camera', [0, -90]])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        script_traj.append(['camera', [0, 90]])
        for _ in range(4):
            script_traj.append(['left', 1])
        script_traj.append(['attack', 1])
        for _ in range(2):  # blank action
            script_traj.append(['camera', [0, 0]])
        script_traj.append(['use', 1])
        for _ in range(4):
            script_traj.append(['forward', 1])
        for _ in range(2):
            script_traj.append(['back', 1])
        for _ in range(2):
            script_traj.append(['right', 1])
        # 8. 给动物建出路
        for _ in range(3):
            for _ in range(5):
                script_traj.append(['right', 1])
        script_traj.append(['camera', [0, -90]])
        for _ in range(30):
            script_traj.append(['attack', 1])
        script_traj.append(['camera', [0, 90]])
        # 9. 向前走，出门
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

        script_actions = []
        for script_transition in script_traj:
            action = deepcopy(NOOP_ACTION)
            action[script_transition[0]] = script_transition[1]
            if len(script_transition) == 4:
                action[script_transition[2]] = script_transition[3]
            if len(script_transition) == 6:
                action[script_transition[2]] = script_transition[3]
                action[script_transition[4]] = script_transition[5]
            script_actions.append(action)

        return script_actions

    def get_pen_traj_v6(self):
        script_traj = []
        # 0. 全局攻击
        script_traj.append(['camera', [0, 0]])
        script_traj.append(['camera', [-15, 0]])
        script_traj.append(['camera', [-10, 0]])
        script_traj.append(['camera', [0, 0]])
        # for _ in range(10):
        #     script_traj.append(['camera', [0, 0]])
        # for _ in range(36):
        #     script_traj.append(['camera', [0, 10]])
        #     script_traj.append(['attack', 1])
        #     script_traj.append(['camera', [0, 0]])
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
        for _ in range(3):
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
        for _ in range(4):
            script_traj.append(['forward', 1])
        for _ in range(2):
            script_traj.append(['back', 1])
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
            script_traj.append(['camera', [0, 30], 'hotbar.9', 1])
        script_traj.append(['use', 1])
        # 11. 后退、抬头
        for _ in range(4):
            script_traj.append(['back', 1])
        script_traj.append(['camera', [-30, 0]])
        # -1.
        for _ in range(5):
            script_traj.append(['camera', [0, 0]])

        script_actions = []
        for script_transition in script_traj:
            action = deepcopy(NOOP_ACTION)
            action[script_transition[0]] = script_transition[1]
            if len(script_transition) == 4:
                action[script_transition[2]] = script_transition[3]
            if len(script_transition) == 6:
                action[script_transition[2]] = script_transition[3]
                action[script_transition[4]] = script_transition[5]
            script_actions.append(action)

        return script_actions

    @property
    def detect_animal_flag(self):
        if len(self.frame_info) >= self.max_frame:
            animal_list = []
            for pred_key in self.frame_info[-1]:
                animal_list.append(self.frame_info[-1][pred_key]['class'])
            animal_set = set(animal_list)
            if len(animal_list) != len(animal_set):
                self.detect_animal_flag_info.append(True)
            else:
                self.detect_animal_flag_info.append(False)
        else:
            self.detect_animal_flag_info.append(False)
        
        if len(self.detect_animal_flag_info) > self.max_frame:
            self.detect_animal_flag_info.pop(0)

        return sum(self.detect_animal_flag_info) == len(self.detect_animal_flag_info)

    # ====================================================================
    #   utils function
    # ====================================================================
    def save_video_record(self, save_path='./demo.mp4'):
        vout = cv2.VideoWriter()
        vout.open(save_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (640, 360), True)

        for img in self.video_record:
            vout.write(img)
        vout.release()
        
    def _human_action_decode(self):
        json_files = []
        with open(self.jsonfile_path, 'r+') as file_obj:
            for line in jsonlines.Reader(file_obj):
                json_files.append(line)
        
        env_actions = []
        for json_file in json_files:
            env_actions.append(self._json_action_to_env_action(json_file))
        
        return env_actions

    def _json_action_to_env_action(self, json_action):
        """
        Converts a json action into a MineRL action.
        Returns (minerl_action, is_null_action)
        """
        # This might be slow...
        env_action = NOOP_ACTION.copy()
        # As a safeguard, make camera action again so we do not override anything
        env_action["camera"] = np.array([0, 0])

        is_null_action = True
        keyboard_keys = json_action["keyboard"]["keys"]
        for key in keyboard_keys:
            # You can have keys that we do not use, so just skip them
            # NOTE in original training code, ESC was removed and replaced with
            #      "inventory" action if GUI was open.
            #      Not doing it here, as BASALT uses ESC to quit the game.
            if key in KEYBOARD_BUTTON_MAPPING:
                env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
                is_null_action = False

        mouse = json_action["mouse"]
        camera_action = env_action["camera"]
        camera_action[0] = mouse["dy"] * CAMERA_SCALER
        camera_action[1] = mouse["dx"] * CAMERA_SCALER

        if mouse["dx"] != 0 or mouse["dy"] != 0:
            is_null_action = False
        else:
            if abs(camera_action[0]) > 180:
                camera_action[0] = 0
            if abs(camera_action[1]) > 180:
                camera_action[1] = 0

        mouse_buttons = mouse["buttons"]
        if 0 in mouse_buttons:
            env_action["attack"] = 1
            is_null_action = False
        if 1 in mouse_buttons:
            env_action["use"] = 1
            is_null_action = False
        if 2 in mouse_buttons:
            env_action["pickItem"] = 1
            is_null_action = False

        env_action['hotbar'] = json_action['hotbar']
        return env_action, is_null_action

    def _get_target_y_angle(self):
        min_delta_angle = float('inf')
        delta_minus_180 = abs(self.angle_y + 180)
        min_delta_angle = min(min_delta_angle, delta_minus_180)
        if min_delta_angle == delta_minus_180:
            self.target_angle[1] = -180
            
        delta_minus_90 = abs(self.angle_y + 90)
        min_delta_angle = min(min_delta_angle, delta_minus_90)
        if min_delta_angle == delta_minus_90:
            self.target_angle[1] = -90

        delta_0 = abs(self.angle_y)
        min_delta_angle = min(min_delta_angle, delta_0)
        if min_delta_angle == delta_0:
            self.target_angle[1] = 0

        delta_90 = abs(self.angle_y - 90)
        min_delta_angle = min(min_delta_angle, delta_90)
        if min_delta_angle == delta_90:
            self.target_angle[1] = 90

        delta_180 = abs(self.angle_y - 180)
        min_delta_angle = min(min_delta_angle, delta_180)
        if min_delta_angle == delta_180:
            self.target_angle[1] = 180
