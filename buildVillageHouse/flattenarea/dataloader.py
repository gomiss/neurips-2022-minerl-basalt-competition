import torch as th
import numpy as np
import torch.utils.data as Data
import os,sys
sys.path.append(os.path.dirname(__file__))
import cv2
from torch.utils.data import DataLoader
import random
from torchvision import datasets, transforms
import torchvision


BASEDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'label_pics')

def allpicpath(r):
    base_dir = BASEDIR
    base_dir = os.path.join(base_dir, str(r))
    if not os.path.exists(base_dir):
        raise ValueError(f"there is no dir {base_dir}")
    pics = os.listdir(base_dir)

    return pics

def picpath2img(p, r):
    base_dir = BASEDIR
    base_dir = os.path.join(base_dir, str(r))
    pic = cv2.imread(os.path.join(base_dir, p))
    return pic



def dataTransforms(frame):
    train_transforms = transforms.Compose([
                                        #    transforms.Resize((224,224)),
                                        transforms.ToTensor(),    
                                        transforms.Resize((224,224)),                            
                                        torchvision.transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225],
        ),
                                        ])
    return np.array(train_transforms(frame))

def PicsDataLoader(labels=[0,1], size=5, batch_size=1, shuffle=True):
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for label in labels:
        
        pics = allpicpath(label)    
        pics_path = [os.path.join(BASEDIR, str(label), v) for v in pics]
        if label == 0:
            pics_path += pics_path
        random.shuffle(pics_path)

        frames = [dataTransforms(cv2.imread(pic_path)) for pic_path in pics_path[:size]]
        train_X = train_X + frames
        train_Y = train_Y + [label for _ in range(size)]

        test_size = int(size*0.2)
        frames = [dataTransforms(cv2.imread(pic_path)) for pic_path in pics_path[size:size+test_size]]
        test_X = test_X + frames
        test_Y = test_Y + [label for _ in range(test_size)]

    # print(train_X)
    train_dataset = Data.TensorDataset(th.tensor(np.array(train_X)), th.tensor(np.array(train_Y)))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    test_dataset = Data.TensorDataset(th.tensor(test_X), th.tensor(test_Y))
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


    return train_data_loader, test_data_loader

if __name__=="__main__":
    train_data_loader, test_data_loader = PicsDataLoader()
    for i, (images, labels) in enumerate(train_data_loader):
        print(labels)
    print(len(train_data_loader), len(test_data_loader))