import os
import cv2
import tqdm
import pickle



def worker(label_dir):
    TRAIN_PATH = os.path.join(label_dir, 'train', 'labels')
    TEST_PATH = os.path.join(label_dir, 'test', 'labels')

    # generate train_file_infos.pkl
    train_file_infos = {}
    for filename in tqdm.tqdm(os.listdir(TRAIN_PATH)):
        video_name, img_index = filename.split('_')
        img_index = int(img_index.split('.')[0])

        if video_name not in train_file_infos:
            train_file_infos[video_name] = {}

        with open(os.path.join(TRAIN_PATH, filename), 'r') as f:
            train_file_infos[video_name][img_index] = {
                'data': f.readlines(),
                'filename': filename.split('.')[0],
            }
    
    with open('train_file_infos.pkl', 'wb') as f:
        pickle.dump(train_file_infos, f)
    
    # generate train_file_infos.pkl
    test_file_infos = {}
    for filename in tqdm.tqdm(os.listdir(TEST_PATH)):
        video_name, img_index = filename.split('_')
        img_index = int(img_index.split('.')[0])

        if video_name not in test_file_infos:
            test_file_infos[video_name] = {}

        with open(os.path.join(TEST_PATH, filename), 'r') as f:
            test_file_infos[video_name][img_index] = {
                'data': f.readlines(),
                'filename': filename.split('.')[0],
            }

    with open('test_file_infos.pkl', 'wb') as f:
        pickle.dump(test_file_infos, f)

def de_worker(file_dir, video_dir, save_path):
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

    TRAIN_PATH = os.path.join(file_dir, 'train_file_infos.pkl')
    TEST_PATH = os.path.join(file_dir, 'test_file_infos.pkl')

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

if __name__ == '__main__':
    DIR = r'C:\Users\zhoutianze\Desktop\crowdsourced_data'
    VIDEO_DIR = r'F:\minerl_data'
    SAVE_PATH = './yolo_data'

    # worker(DIR)
    de_worker('./', VIDEO_DIR, SAVE_PATH)

