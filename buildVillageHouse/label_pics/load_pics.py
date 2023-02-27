import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys,os,glob
from collections import defaultdict
import json


def read_json(json_path):
    with open(json_path, 'r',encoding='utf-8') as fp:
        data = json.load(fp)
    return data

def load_pics(json_data, video_path, output_path):
    data = json_data
    for video, tick_label_list in data.items():
        ticks = [int(t[0]) for t in tick_label_list]
        labels = [str(t[1]) for t in tick_label_list]
        max_tick = ticks[-1] + 1
        p = 0
        if not os.path.exists(os.path.join(video_path, video)):
            continue
        reader = cv.VideoCapture(os.path.join(video_path, video))
        for cur_tick in range(max_tick):
            # print(f"{video}_{cur_tick}")
            ret, frame = reader.read()
            if cur_tick == ticks[p]:
                img = frame
                cv.imwrite(os.path.join(os.path.join(output_path,labels[p]), video + "_" + str(cur_tick) + ".png"), img)
                p += 1
                if p == len(ticks):
                    break
        reader.release()

def load_pics_main():

    base_dir = os.path.dirname(os.path.dirname(__file__))
    json_path = os.path.join(base_dir,'label_pics', 'resnet50label.json')
    json_data = read_json(json_path)
    video_dir = os.path.join(os.path.dirname(base_dir), 'data', 'MineRLBasaltBuildVillageHouse-v0')
    output_dir = os.path.dirname(__file__)
    # print(f"base_dir: {base_dir}, viode_dir: {video_dir}, output_dir: {output_dir}")
    load_pics(json_data, video_dir, output_dir)

if __name__=="__main__":
    load_pics_main()