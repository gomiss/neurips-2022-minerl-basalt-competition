import os
import cv2
import time
import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yolov5'))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression


class AnimalDetector():
    def __init__(self, 
                 weights,
                 data,
                 conf_thres=0.25,
                 iou_thres=0.45,
                 agnostic_nms=False):
        self.weights = weights
        self.data = data
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic_nms = agnostic_nms
        self.animal_class = ['cliff']

        self.init_model()

    def init_model(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('=====================================')
        print(f'detector model: {self.device}')
        print('=====================================')
        self.model = DetectMultiBackend(self.weights, 
                                        device=self.device, 
                                        dnn=False, 
                                        data=self.data, 
                                        fp16=False)
        return self.device

    def inference(self, img):
        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
        img = img[None]
        # img = img[..., ::-1].transpose((0, 3, 1, 2))
        img = img.transpose((0, 3, 1, 2))
        img = torch.from_numpy(img.copy()).to(self.device)
        img = img.float()
        img = img / 255
        
        res = {}
        with torch.no_grad():
            # 1. inference
            pred = self.model(img)
            
            # 2. NMS
            pred = non_max_suppression(pred, 
                                       self.conf_thres,
                                       self.iou_thres,
                                       None,
                                       self.agnostic_nms,
                                       max_det=1000)[0]

            # 3. pred handle
            pred = pred.cpu().numpy()
            n_pred = pred.shape[0]
            for idx in range(n_pred):
                if int(pred[idx][5]) == 2: # no detect flower
                    continue

                res[idx] = {
                    'pos': [
                        (pred[idx][0] + pred[idx][2]) / 2, 
                        (pred[idx][1] + pred[idx][3]) / 2
                    ],
                    'size': [
                        pred[idx][2] - pred[idx][0],
                        pred[idx][3] - pred[idx][1]
                    ],
                    'conf': pred[idx][4],
                    'class': self.animal_class[int(pred[idx][5])]
                }
            
        return res

    def check(self, img):
        res = self.inference(img)
        self.visualize(img, res)

        if len(res) == 0 or res[0]["conf"] < 0.8:
            return False

        if len(res) > 0:
            img, _ = self.cv_handle(img, {0: res[0]})
            print(res)
            plt.imsave('tmp/terminal_conf{:.2f}_{}.png'.format(res[0]["conf"], time.time()), img)
        return True

    def cv_handle(self, img, yolo_res):
        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)

        handled_yolo_res = {}
        for key in yolo_res:
            _pos = yolo_res[key]['pos']
            _size = yolo_res[key]['size']
            _conf = yolo_res[key]['conf']
            _class = yolo_res[key]['class']

            pos_top_left = (int(_pos[0] - _size[0] / 2), int(_pos[1] - _size[1] / 2))
            pos_bottom_right = (int(_pos[0] + _size[0] / 2), int(_pos[1] + _size[1] / 2))
            cv2.rectangle(img, pos_top_left, pos_bottom_right, (255, 255, 0), 3)
            cv2.putText(img=img, text='{:.2f}'.format(_conf), org=(pos_top_left[0], pos_top_left[1] - 5),
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale=0.8, color=(255, 255, 0), thickness=3)

            handled_yolo_res[key] = {
                'pos': _pos,
                'size': _size,
                'conf': _conf,
                'class': _class
            }


        # img = img[:, :, ::-1]
        img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_LINEAR)
        # plt.imsave('tmp/1.png'.format(res[0]["conf"], time.time()), img)
        return img, handled_yolo_res

    def visualize(self, img, res):
        img, _ = self.cv_handle(img, res)
        cv2.imshow('run', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    IMG_PATH = 'yolo_traindata_v1/valid/images/cheeky-cornflower-setter-00df0a566647-20220717-135630-25_png.rf.099fa69f58238a3802f822e1782b6b83.jpg'
    # IMG_PATH = 'test1.png'
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # animal_detector = AnimalDetector('../train/yolo_300/best.pt', 'data.yaml')
    animal_detector = AnimalDetector('../yolov5/runs/train/exp2/weights/best.pt', 'data.yaml')

    res = animal_detector.inference(img)
    print(res)
    if len(res) > 0:
        animal_detector.cv_handle(img, {0: res[0]})