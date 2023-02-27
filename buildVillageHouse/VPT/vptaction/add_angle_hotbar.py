import numpy as np
import pandas as pd
from tqdm import tqdm
import os, json
import os,sys
sys.path.append(os.path.dirname(__file__))


from actionAgent import ActionAgent

vdf = pd.read_csv("video_list.csv")
video_list = [list(x[~pd.isnull(x)]) for x in np.array(vdf[["0","1","2"]])]


class ToWardsState:
    def __init__(self) -> None:

        self.YMAX = 180
        self.XMAX = 360

        self.x = 90.0
        self.y = 90.0
        
    def mouse_move(self, dx, dy):
        self.x = (self.x + dx) % self.XMAX

        self.y = min(max(0, (self.y + dy)), self.YMAX)

    def __repr__(self) -> str:
        return f"x: {self.x}, y: {self.y}"

dataset_dir = "/data/MineRLBasaltBuildVillageHouse-v0"
actionAgent = ActionAgent()

print(len(video_list))
print(video_list[277])

for i in tqdm(range(len(video_list))):

    towardstate = ToWardsState()

    for j in range(len(video_list[i])):


        try:
            uid = video_list[i][j].split(".")[0]
        except:
            print(f"error with {video_list[i]}, j: {j}")

        extra_states = []
        json_path = os.path.join(dataset_dir, uid+".jsonl")
        env_action_list = actionAgent.json_action_file_data_loader_preprocess(json_path)

        extra_state = {"angle":(towardstate.x, towardstate.y)}
        extra_states.append(extra_state)

        for ac in env_action_list:

            dy, dx = ac['camera']
            towardstate.mouse_move(dx,dy)

            extra_state = {"angle":(towardstate.x, towardstate.y)}
            extra_states.append(extra_state)



        out_put_dir = "/minerl/extrastate"
        OUTPUTFILE = os.path.join(out_put_dir, video_list[i][j].split(".")[0] + "_extraState.jsonl")

        with open(OUTPUTFILE, 'w') as json_file:
            for k in range(len(extra_states)):
                json_line = extra_states[k]
                json.dump(json_line, json_file)
                json_file.write("\n")

        # print(OUTPUTFILE)
