import os
import cv2
import numpy as np

base_path = r'E:\RL\minecraft\submit\neurips-2022-minerl-basalt-competition\test10_avoid_water'
for i in range(20):
    print(i)
    path = os.path.join(base_path, 'debug_sequence' + str(i))
    # if os.path.exists(os.path.join(path, '3599.png')):
    #     print(path)
    # continue
    filelist = os.listdir(path)

    fps = 24 #视频每秒24帧
    size = (640, 360) #需要转为视频的图片的尺寸
    #可以使用cv2.resize()进行修改
    if os.path.exists(os.path.join(path,"Video.mp4")):
        os.remove(os.path.join(path,"Video.mp4"))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(base_path,"test_case10Video" + str(i)+ ".mp4"),fourcc, fps, size)
    #视频保存在当前目录下
    max_int = -1
    for item in filelist:
        if item.endswith('.png'):
            if int(item.split('.')[0]) > max_int:
                max_int = int(item.split('.')[0])
    for j in range(max_int+1):
        item = os.path.join(path, str(j)+'.png')
        img = cv2.imread(item)
        video.write(img)
    # for item in filelist:
    #     if item.endswith('.png'):
    #     #找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
    #         item = os.path.join(path,item)
    #         img = cv2.imread(item)
    #         video.write(img)

    video.release()
    cv2.destroyAllWindows()


