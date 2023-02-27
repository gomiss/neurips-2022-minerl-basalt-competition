import gym
import minerl
import numpy as np
import time
import json
import random
import cv2 as cv
import os
from collections import OrderedDict
import os,sys
sys.path.append(os.path.dirname(__file__))
from label_pics.load_pics import load_pics_main
from flattenarea.train_buildvillagehouse_resnet50 import train_flatten_area
from VPT.train_bulidvillagehouse_vpt import train_bc

def move_models():
    base_dir = os.path.dirname(__file__)
    model_dir = os.path.join(base_dir, "models")
    train_dir = os.path.join(os.path.dirname(base_dir), "train")
    os.system(f"""mv {os.path.join(train_dir,"flattenareaV5mobilenet.pt")} {model_dir}  && mv {os.path.join(train_dir, "MineRLBasaltMovingNoinv.weights")} {model_dir}""")

def train_resnet():
    load_pics_main()
    train_flatten_area()


def train_VPT():
    train_bc()

def train_buildvillagehouse_main():
    train_resnet()
    train_VPT()
    move_models()

if __name__ == '__main__':
    train_buildvillagehouse_main()