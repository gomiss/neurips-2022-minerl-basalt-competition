import os
import pickle
import time

import numpy as np
import cv2 as cv
import glob

def process(file_name):
    cap = cv.VideoCapture(file_name)
    i = 1
    save_path = f"{os.path.dirname(file_name)}/image"
    save_name = os.path.basename(file_name).split('.')[0]
    print(f"")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    while cap.isOpened():
        ret, frame = cap.read()
        # 如果正确读取帧，ret为True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # cv.imshow('frame', gray)
        if i % 5 == 0:
            cv.imwrite('{}/{}-{}.png'.format(save_path, save_name, i), frame)
        if cv.waitKey(1) == ord('q'):
            break
        i += 1
    cap.release()
    cv.destroyAllWindows()

def walk_through(dir_path):
    for file_name in glob.glob(f"{dir_path}/*.mp4")[:100]:
        process(file_name)

def process_idx(file_name, save_path, frame_idxes):
    cap = cv.VideoCapture(file_name)
    save_name = os.path.basename(file_name).split('.')[0]
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for frame_idx in frame_idxes:
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx-1)
        ret, frame = cap.read()
        cv.imwrite('{}/{}-{}.png'.format(save_path, save_name, frame_idx), frame)

    cap.release()
    cv.destroyAllWindows()


def get_classification_data(data_path):
    """Extract data from the given videos"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    video_info_path = os.path.join(base_dir, 'waterfall/waterfall_video_info.pkl')

    with open(video_info_path, 'rb') as f:
        video_data = pickle.load(f)

    for save_path, data in video_data.items():
        for video_name, frames in data.items():
            process_idx(os.path.join(data_path, video_name),
                        save_path=os.path.join(base_dir, 'waterfall', save_path),  # waterfall/waterfall_classify_data
                        frame_idxes=frames)
        print(f"finish process {save_path}")
    print("Data all finished")


if __name__ == "__main__":
    # walk_through("/mnt/d/project/basalt-2022-behavioural-cloning-baseline/utils/MineRLBasaltMakeWaterfall-v0")
    get_classification_data(data_path='/mnt/d/project/basalt-2022-behavioural-cloning-baseline/utils/MineRLBasaltMakeWaterfall-v0/')