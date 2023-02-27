import os
import cv2
import tqdm
import pickle
import shutil


def de_worker(file_dir, video_dir, save_path, yolo_data_path):
    def convert_video_to_imgs(video_path):
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
        return imgs

    TRAIN_PATH = os.path.join(file_dir, 'animalpen_train_file_infos.pkl')
    TEST_PATH = os.path.join(file_dir, 'animalpen_test_file_infos.pkl')

    # get label data
    with open(TRAIN_PATH, 'rb') as f:
        train_file_infos = pickle.load(f)
    with open(TEST_PATH, 'rb') as f:
        test_file_infos = pickle.load(f)

    # mkdir
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(os.path.join(save_path, 'train')):
        os.mkdir(os.path.join(save_path, 'train'))
    if not os.path.exists(os.path.join(save_path, 'test')):
        os.mkdir(os.path.join(save_path, 'test'))
    if not os.path.exists(os.path.join(save_path, 'valid')):
        os.mkdir(os.path.join(save_path, 'valid'))
    if not os.path.exists(os.path.join(save_path, 'train', 'images')):
        os.mkdir(os.path.join(save_path, 'train', 'images'))
    if not os.path.exists(os.path.join(save_path, 'train', 'labels')):
        os.mkdir(os.path.join(save_path, 'train', 'labels'))
    if not os.path.exists(os.path.join(save_path, 'test', 'images')):
        os.mkdir(os.path.join(save_path, 'test', 'images'))
    if not os.path.exists(os.path.join(save_path, 'test', 'labels')):
        os.mkdir(os.path.join(save_path, 'test', 'labels'))
    if not os.path.exists(os.path.join(save_path, 'valid', 'images')):
        os.mkdir(os.path.join(save_path, 'valid', 'images'))
    if not os.path.exists(os.path.join(save_path, 'valid', 'labels')):
        os.mkdir(os.path.join(save_path, 'valid', 'labels'))

    # =============  data convert =============
    video_dir = os.path.join(video_dir, 'MineRLBasaltCreateVillageAnimalPen-v0')
    
    # train data convert
    for video_name in tqdm.tqdm(train_file_infos):
        video_path = os.path.join(video_dir, video_name+'.mp4')
        video_imgs = convert_video_to_imgs(video_path)

        # save imgs
        for img_index in train_file_infos[video_name]:
            cv2.imwrite(
                os.path.join(save_path, 'train', 'images', train_file_infos[video_name][img_index]['filename']+'.jpg'), 
                video_imgs[img_index]
            )

        # save labels
        for img_index in train_file_infos[video_name]:
            with open(os.path.join(save_path, 'train', 'labels', train_file_infos[video_name][img_index]['filename']+'.txt'), 'w') as f:
                for info in train_file_infos[video_name][img_index]['data']:
                    f.write(info)
    
    # test data convert
    for video_name in tqdm.tqdm(test_file_infos):
        video_path = os.path.join(video_dir, video_name+'.mp4')
        video_imgs = convert_video_to_imgs(video_path)

        # save imgs
        for img_index in test_file_infos[video_name]:
            cv2.imwrite(
                os.path.join(save_path, 'test', 'images', test_file_infos[video_name][img_index]['filename']+'.jpg'), 
                video_imgs[img_index]
            )

        # save labels
        for img_index in test_file_infos[video_name]:
            with open(os.path.join(save_path, 'test', 'labels', test_file_infos[video_name][img_index]['filename']+'.txt'), 'w') as f:
                for info in test_file_infos[video_name][img_index]['data']:
                    f.write(info)
    
    # valid data convert
    for video_name in tqdm.tqdm(test_file_infos):
        video_path = os.path.join(video_dir, video_name+'.mp4')
        video_imgs = convert_video_to_imgs(video_path)

        # save imgs
        for img_index in test_file_infos[video_name]:
            cv2.imwrite(
                os.path.join(save_path, 'valid', 'images', test_file_infos[video_name][img_index]['filename']+'.jpg'), 
                video_imgs[img_index]
            )

        # save labels
        for img_index in test_file_infos[video_name]:
            with open(os.path.join(save_path, 'valid', 'labels', test_file_infos[video_name][img_index]['filename']+'.txt'), 'w') as f:
                for info in test_file_infos[video_name][img_index]['data']:
                    f.write(info)
    
    # yaml file
    with open(os.path.join(save_path, 'data.yaml'), 'w') as f:
        f.write(f'train: {yolo_data_path}/train/images\n')
        f.write(f'val: {yolo_data_path}/valid/images\n')
        f.write(f'test: {yolo_data_path}/test/images\n')
        f.write('\n')
        f.write('nc: 7\n')
        f.write('names: [\'cow\', \'duck\', \'flower\', \'people\', \'pig\', \'rabbit\', \'sheep\']')


def train_animalpen(
    FILE_DIR='./yolov5/resources/',
    VIDEO_DIR='/data/',
    SAVE_PATH ='./yolov5/data/animalpen_data/',
    TRAIN_YOLO_SH_PATH='./yolov5/train_animal_detector.sh',
    ORIGINAL_YOLO_PT_PATH='./yolov5/runs/train/exp/weights/best.pt',
    TARGET_YOLO_PT_PATH='./train/animal_detector.pt'
):
    base_dir = os.path.abspath(__file__)
    base_dir = base_dir[:base_dir.rindex('/')]
    YOLO_DATA_PATH = os.path.join(base_dir, 'data', 'animalpen_data')

    # 1. 解码yolo数据集 [i7-8700 32G]: 570.76s
    de_worker(FILE_DIR, VIDEO_DIR, SAVE_PATH, YOLO_DATA_PATH)

    # 2. 训练yolo模型
    f = os.popen(TRAIN_YOLO_SH_PATH)
    print(f.readlines())
    shutil.copyfile(
        ORIGINAL_YOLO_PT_PATH,
        TARGET_YOLO_PT_PATH
    )
    shutil.copyfile(
        './yolov5/data/animalpen_data/data.yaml',
        './train/animal_detector.yaml'
    )


if __name__ == '__main__':
    FILE_DIR = './resources/'
    VIDEO_DIR = '/data/'
    SAVE_PATH = './yolov5/data/animalpen_data'

    base_dir = os.path.abspath(__file__)
    base_dir = base_dir[:base_dir.rindex('/')]
    YOLO_DATA_PATH = os.path.join(base_dir, 'yolov5', 'data', 'animalpen_data')

    # 1. 解码yolo数据集 [i7-8700 32G]: 570.76s
    de_worker(FILE_DIR, VIDEO_DIR, SAVE_PATH, YOLO_DATA_PATH)

    # 2. 训练yolo模型
    f = os.popen('./yolov5/train_animal_detector.sh')
    print(f.readlines())
    shutil.copyfile(
        './yolov5/runs/train/exp/weights/best.pt',
        './train/animal_detector.pt'
    )

