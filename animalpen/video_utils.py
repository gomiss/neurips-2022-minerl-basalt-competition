import os
import cv2
import random
import shutil
import jsonlines
import multiprocessing

import numpy as np


# =======================================================================
#     Utils Function
# =======================================================================
def convert_video_to_imgs(video_path, dropout_first_frame=True):
    # read video
    cap = cv2.VideoCapture(video_path)    
    flag = cap.isOpened()
    if not flag:
        print("\033[31mLine 65 error\033[31m: open" + video_path + "error!")

    imgs = []
    while True:
        flag, frame = cap.read()
        if not flag:  # if last frame, return
            break
        
        imgs.append(frame)
    
    cap.release()

    if dropout_first_frame:
        return imgs[1:]
    else:
        return imgs


def save_imgs_to_video(imgs, save_path, fps=30, size=(640, 360)):
    vout = cv2.VideoWriter()
    vout.open(save_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size, True)

    for img in imgs:
        vout.write(img)
    vout.release()





