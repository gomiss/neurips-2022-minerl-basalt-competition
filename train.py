# Train one model for each task
from behavioural_cloning import behavioural_cloning_train
from waterfall.classify import train_epoch
from yolov5.train_animalpen_yolo import train_animalpen
from yolov5.train_findcave_yolo import train_findcave
import os,sys
os.environ['TORCH_HOME'] = os.path.dirname(__file__)
sys.path.append(os.path.dirname(__file__))
from buildVillageHouse.train_buildvillagehouse import train_buildvillagehouse_main


def main():
    
    print("===Training MakeWaterfall model===")
    train_epoch()

    # train yolo
    os.system('chmod a+x ./yolov5/train_animal_detector.sh')
    os.system('chmod a+x ./yolov5/train_findcave.sh')
    os.system(f'''cp hub/checkpoints/yolov5s.pt train/''')
    print("===Training yolo models===")
    train_animalpen()
    train_findcave()

    print("===Training BuildVillageHouse model===")
    train_buildvillagehouse_main()

    print("===Training FindCave model===")
    behavioural_cloning_train(
        data_dir="data/MineRLBasaltFindCave-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltFindCave.weights"
    )


if __name__ == "__main__":
    main()
