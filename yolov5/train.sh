# python train.py --img 640 --batch 16 --epochs 50 --data ./data/minecraft_original/data.yaml --weights yolov5m.pt
# python train.py --img 640 --batch 16 --epochs 100 --data ./data/minecraft_with_flower/data.yaml --weights yolov5m.pt
# python train.py --img 640 --batch 16 --epochs 50 --data ./data/minecraft_with_flower_people_v2/data.yaml --weights yolov5m.pt
# python train.py --img 640 --batch 16 --epochs 50 --data ./data/crowdsourced_data/data.yaml --weights yolov5m.pt
#python train.py --img 640 --batch 16 --epochs 500 --data ./data/dest/data.yaml --weights yolov5m.pt
python train.py --img 640 --batch 16 --epochs 100 --data ./data/yolo_data/data.yaml --weights yolov5m.pt
