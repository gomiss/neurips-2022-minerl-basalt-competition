import random
import os
import shutil
import numpy as np
import cv2

from collections import OrderedDict
NOOP_ACTION = OrderedDict([('ESC', 0),
             ('attack', 0),
             ('back', 0),
             ('camera', np.array([0., 0.], dtype=np.float32)),
             ('drop', 0),
             ('forward', 0),
             ('hotbar.1', 0),
             ('hotbar.2', 0),
             ('hotbar.3', 0),
             ('hotbar.4', 0),
             ('hotbar.5', 0),
             ('hotbar.6', 0),
             ('hotbar.7', 0),
             ('hotbar.8', 0),
             ('hotbar.9', 0),
             ('inventory', 0),
             ('jump', 0),
             ('left', 0),
             ('pickItem', 0),
             ('right', 0),
             ('sneak', 0),
             ('sprint', 0),
             ('swapHands', 0),
             ('use', 0)])

def waterfall_data():
    from_dir = r"D:\project\basalt-2022-behavioural-cloning-baseline\utils\MineRLBasaltMakeWaterfall-v0\image"
    to_dir = r"F:\project\neurips-2022-minerl-basalt-competition\waterfall\waterfall_classify_data\train"

    exist_names = os.listdir(os.path.join(to_dir, 'cliff'))
    names = os.listdir(from_dir)[:10000]
    i = 0
    while True:
        n = random.choice(names)
        if n not in exist_names:
            exist_names.append(n)
            i += 1
            shutil.copy2(os.path.join(from_dir, n), os.path.join(to_dir, 'none'))

        if i >= 1500:
            break


def zip_folder(size=28):
    os.system(f"cd hub && 7z a -v{size}m myfiles.zip checkpoints")

def unzip(base_dir):
    os.system(f"""cd {base_dir}/hub && 7z x myfiles.zip.001 -y && 
            mv checkpoints/resnet_500_best.pb {os.path.join(base_dir, 'train/')}
            """)

def get_noop_action():
    return NOOP_ACTION.copy()

class BubbleCheck:
    def __init__(self, template_path, mask_path=None):
        self.template = cv2.imread(template_path, 0)
        self.mask = None if mask_path is None else cv2.imread(mask_path, 0)

    def check(self, img):
        img = img[311:320, 402:411, :]
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        res = cv2.matchTemplate(img_gray, self.template, cv2.TM_CCOEFF_NORMED, mask=self.mask)

        threshold = 0.6
        loc = np.where(res >= threshold)

        # w, h = self.template.shape[::-1]
        # for pt in zip(*loc[::-1]):
        #     cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        # cv2.imwrite("r.png", img)
        if len(loc[0]) > 0:
            return True
        else:
            return False


if __name__ == "__main__":
    zip_folder()