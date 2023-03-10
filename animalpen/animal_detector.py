import os
import cv2
import time
import torch
import torchvision

import numpy as np

from animalpen.models.common import DetectMultiBackend


def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.3 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

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
        self.animal_class = ['cow', 'duck', 'flower', 'people', 'pig', 'rabbit', 'sheep']

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
        #img = img[..., ::-1].transpose((0, 3, 1, 2))
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



if __name__ == '__main__':
    IMG_PATH = './resources/845.jpg'
    img = cv2.imread(IMG_PATH)

    animal_detector = AnimalDetector('./resources/best_m.pt','./resources/data.yaml')

    res = animal_detector.inference(img)
    print(res)