import cv2, json, os, glob
from tqdm import tqdm
import pandas as pd
import numpy as np



data_dir = "/data/MineRLBasaltBuildVillageHouse-v0"

vdf = pd.read_csv("/minerl/basalt-2022-behavioural-cloning-baseline/action/video_list.csv")
x = np.array(vdf[["0","1","2"]])[0]
list(x[~pd.isnull(x)])
video_list = [list(x[~pd.isnull(x)]) for x in np.array(vdf[["0","1","2"]])]
# print(video_list)

unique_ids = [v[0] for v in video_list]
unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))

demo_tuples = []

# print(unique_ids)
for uid in unique_ids:

    video_path = os.path.abspath(os.path.join(data_dir, uid + ".mp4"))
    json_path = os.path.abspath(os.path.join(data_dir, uid + ".jsonl"))
    demo_tuples.append((video_path, json_path))

# print(len(demo_tuple))
# print(demo_tuple[0])

for demo_tuple in tqdm(demo_tuples):
    video_path, json_path = demo_tuple
    with open(json_path) as json_file:
        json_lines = json_file.readlines()
        json_data = "[" + ",".join(json_lines) + "]"
        json_data = json.loads(json_data)
    for i in range(len(json_data)):
        step_data = json_data[i]
        if step_data["mouse"]["newButtons"] == [1]:
            break

    e_p = i
    
    reader = cv2.VideoCapture(video_path)
    ret, frame = reader.read()

    base_dir = "/minerl/basalt-2022-behavioural-cloning-baseline/clipVideo"
    OUTPUT_FILE = os.path.join(base_dir, video_path.split("/")[-1].split(".")[0] + f"_{0}_{e_p}.mp4") 
    writer = cv2.VideoWriter(OUTPUT_FILE,
                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), # type - mp4
                20, # fps
                (640, 360)) # resolution
    
    for wp in range(e_p):
        ret, frame = reader.read()
        writer.write(frame)

    writer.release()
    reader.release()

    base_dir = "/minerl/basalt-2022-behavioural-cloning-baseline/clipVideo"
    OUTPUT_FILE =  os.path.join(base_dir, json_path.split("/")[-1].split(".")[0] + f"_{0}_{e_p}.jsonl")

    with open(OUTPUT_FILE, 'w') as json_file:
        for i in range(e_p):
            json_line = json_data[i]
            json.dump(json_line, json_file)
            json_file.write("\n")

    






