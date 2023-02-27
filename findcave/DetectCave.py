import queue
from operator import itemgetter, attrgetter
from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)
import os
import numpy as np
import copy
import torch
from findcave.config import IMG_WIDTH, IMG_HEIGHT, LOCAL_DEDUG




class DetectCave:
    def __init__(self, device, weights_path):
        self.model = self.init_model(device, weights_path)
        self.reset()

    def init_model(self, device, weights_path):
        # weights_path = os.path.join(ROOT, r'runs\train\exp13\weights\best.pt')
        model = DetectMultiBackend(weights_path, device=device, dnn=False, fp16=False)
        return model

    def reset(self):
        self.has_found_cave = [False for i in range(10)]
        self.list_idx = 0

    def judge_in_cave(self, image, detect_result):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        r, binary = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
        # if 'cave' in detect_result and len(detect_result['cave'])>0:
        #     rect_left_up = (detect_result['cave'][0][0], detect_result['cave'][0][1])
        #     rect_right_down = (detect_result['cave'][0][2], detect_result['cave'][0][3])
        #     cv2.imshow("cave_res", image[rect_left_up[1]:rect_right_down[1],rect_left_up[0]:rect_right_down[0], :])
        #     cv2.waitKey()
        # exclude deep sea
        lower = np.array([100, 43, 20])
        upper = np.array([124, 255, 255])
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        bluemask = cv2.inRange(hsv, lower, upper)
        binary[(bluemask == 255)] = 255
        lower = np.array([35, 43, 20])
        upper = np.array([77, 255, 255])

        greenmask = cv2.inRange(hsv, lower, upper)
        binary[(greenmask == 255)] = 255

        lower = np.array([15, 128, 16])
        upper = np.array([21, 216, 52])
        treemask = cv2.inRange(hsv, lower, upper)
        binary[(treemask == 255)] = 255

        binary[260:, 406:] = 0
        binary[320:, 225:406] = 0
        static_mask_area = 640 * 360 - (360 - 260) * (640 - 406) + (360 - 320) * (406 - 225)

        if np.sum(binary) / 255 / static_mask_area < 0.12:
            self.has_found_cave[self.list_idx % 10] = True
            print("blackblackblackblackblackblackblackblackblackblackblackblack")
        else:
            if 'cave' in detect_result and len(detect_result['cave']) > 0:
                rect_left_up = (detect_result['cave'][0][0], detect_result['cave'][0][1])
                rect_right_down = (detect_result['cave'][0][2], detect_result['cave'][0][3])
                cave_area = (rect_right_down[1] - rect_left_up[1]) * (rect_right_down[0] - rect_left_up[0])
                if LOCAL_DEDUG:
                    print("cave_area: ", cave_area)
                if cave_area / IMG_WIDTH /IMG_HEIGHT > 0.13 and \
                        np.sum(binary[rect_left_up[1]:rect_right_down[1],rect_left_up[0]:rect_right_down[0]])/ 255/cave_area > 0.1:

                    self.has_found_cave[self.list_idx % 10] = True
                else:
                    self.has_found_cave[self.list_idx % 10] = False
            else:
                self.has_found_cave[self.list_idx % 10] = False
        self.list_idx = (self.list_idx+1) % 10
        if LOCAL_DEDUG:
            print('hasfound: ', self.has_found_cave, self.list_idx)
            if sum(self.has_found_cave) > 6:
                print("terminalterminalterminalterminalterminalterminalterminal")
        return sum(self.has_found_cave) > 6

    def detect_image(self, image, device, step, now_i):
        image = np.ascontiguousarray(image)
        im0 = copy.deepcopy(image)
        ori_im = copy.deepcopy(image)

        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = (640, 640)
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        bs = 1
        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

        stride = 32
        auto = True
        im = letterbox(im0, imgsz, stride=stride, auto=auto)[0]  # padded resize
        # im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = im.transpose((2, 0, 1))  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        color = (255, 56, 56)
        txt_color = (255, 255, 255)
        lw = 3
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = self.model(im)

        # NMS
        with dt[2]:
            conf_thres, iou_thres, classes, agnostic_nms, max_det = 0.25, 0.45, None, False, 1000
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        result = {}

        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for *box, conf, cls in reversed(det):
                if names[int(cls)] == 'cave' and conf < 0.4:
                    continue
                if names[int(cls)] == 'hole' and conf < 0.6:
                    continue

                lower = np.array([100, 43, 46])
                upper = np.array([124, 255, 255])
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                bluemask = cv2.inRange(hsv, lower, upper)
                # cv2.imwrite('bluemask.png', bluemask[int(box[1]): int(box[3]), int(box[0]): int(box[2])])
                blue_rate = np.mean(bluemask[int(box[1]): int(box[3]), int(box[0]): int(box[2])]) / 255
                if LOCAL_DEDUG:
                    print('blue_rate: ',blue_rate)
                if blue_rate > 0.01:
                    continue

                lower = np.array([8, 202, 100])
                upper = np.array([16, 255, 240])

                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                magma_mask = cv2.inRange(hsv, lower, upper)
                magma_rate = np.sum(magma_mask[int(box[1]): int(box[3]), int(box[0]): int(box[2])]) / int(
                    box[3] - box[1]) / int(box[2] - box[0]) / 255
                # magma_percent = np.sum(magma == 255) / magma.size
                if magma_rate > 0.01:
                    continue

                c = int(cls)  # integer class
                if LOCAL_DEDUG:
                    label = f'{names[c]} {conf:.2f}'
                    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    print(p1, p2)
                    # cv2.imwrite("222.png", image)
                if names[c] in result.keys():
                    result[names[c]].append([int(box[0]), int(box[1]), int(box[2]), int(box[3]), abs(int(box[0])+int(box[2])-IMG_WIDTH/2), float(conf), names[c]])
                else:
                    result[names[c]] = [[int(box[0]), int(box[1]), int(box[2]), int(box[3]), abs(int(box[0])+int(box[2])-IMG_WIDTH/2),float(conf), names[c]]]

                if LOCAL_DEDUG:
                    cv2.rectangle(image, p1, p2, color, thickness= lw, lineType=cv2.LINE_AA)
                    tf = max(3 - 1, 1)  # font thickness
                    w, h = cv2.getTextSize(label, 0, fontScale= lw / 3, thickness=tf)[0]  # text width, height
                    outside = p1[1] - h >= 3
                    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                                0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
            if LOCAL_DEDUG:
                if len(det):
                    # cv2.imshow('yolo', image)
                    cv2.imwrite("debug_sequence" +str(now_i) + "/" + str(step) + ".png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for k in result.keys():
            result[k].sort(key=itemgetter(4),reverse=True)
        if LOCAL_DEDUG:
            print(result)
        return result, self.judge_in_cave(ori_im, result)



