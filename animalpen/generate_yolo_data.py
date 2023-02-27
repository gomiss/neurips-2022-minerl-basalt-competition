import os
import cv2
import tqdm

from animal_detector import AnimalDetector
from video_utils import convert_video_to_imgs


def worker(video_path, save_path):
    animal_detector = AnimalDetector('./resources/best_m.pt','./resources/data.yaml')
    imgs = convert_video_to_imgs(video_path)
    n_imgs = len(imgs)
    
    # get yolo detect image index
    useful_img_indexes = []
    for idx, img in enumerate(tqdm.tqdm(imgs)):
        yolo_res = animal_detector.inference(img)
        if len(yolo_res) > 0:
            useful_img_indexes.append(idx)

    # expends useful_imgs
    first_index = True
    expends_img_indexes = []
    for idx in useful_img_indexes:
        if first_index:
            start_index = max(0, idx - 40)
            end_index = min(idx + 5, n_imgs)

            expends_img_indexes.extend([i for i in range(start_index, end_index)])
            first_index = False
        else:
            start_index = max(expends_img_indexes[-1], idx - 5)
            end_index = min(idx + 5, n_imgs)

            expends_img_indexes.extend([i for i in range(start_index, end_index)])
    
    expends_img_indexes = list(set(expends_img_indexes))
    expends_img_indexes.sort()
    
    # save imgs
    img_prefix = video_path.split('\\')[-1].split('.')[0]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for idx in expends_img_indexes:
        cv2.imwrite(save_path + '/{}_{}.jpg'.format(img_prefix, idx), imgs[idx])

if __name__ == '__main__':
    video_path = r'F:\minerl_data\penanimalvideo\cow\wiggy-aquamarine-tapir-4af2568065c6-20220723-085555.mp4'
    save_path = r'F:\minerl_data\penanimalvideo\cow\wiggy-aquamarine-tapir-4af2568065c6-20220723-085555'
    worker(video_path, save_path)


