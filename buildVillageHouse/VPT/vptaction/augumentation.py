from actionAgent import ActionAgent
import cv2, json, os, glob
from tqdm import tqdm


def gen_agent_button_action_list(json_path):
    actionAgent = ActionAgent()
    env_action_list = actionAgent.json_action_file_data_loader_preprocess(json_path)
    agent_button_action_list = []
    for i, env_action in enumerate(env_action_list):
        policy_action = actionAgent.action_transformer.env2policy(env_action)
        if policy_action["camera"].ndim == 1:
            policy_action = {k: v[None] for k, v in policy_action.items()}
        agent_action = actionAgent.env2agent_action_map.from_factored(policy_action)
        agent_button_action = actionAgent.env2agent_action_map.BUTTONS_IDX_TO_COMBINATION[int(agent_action['buttons'].squeeze())]
        agent_button_action_list.append((agent_action, agent_button_action))

    return agent_button_action_list

def gen_video(video_path, augument_list):

    reader = cv2.VideoCapture(video_path)

    ret, frame = reader.read()

    p = 0

    for augu in augument_list:

        s_p, e_p = augu
        base_dir = "/minerl/basalt-2022-behavioural-cloning-baseline/augumenteVideo"
        OUTPUT_FILE = os.path.join(base_dir, video_path.split("/")[-1].split(".")[0] + f"_{s_p}_{e_p}.mp4") 
        # print("OUTPUT_FILE", OUTPUT_FILE)
        writer = cv2.VideoWriter(OUTPUT_FILE,
                        cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), # type - mp4
                        20, # fps
                        (640, 360)) # resolution

        

        for _ in range(s_p - p):
            ret, frame = reader.read()

        for wp in range(e_p - s_p):
            ret, frame = reader.read()
            writer.write(frame)

        p = e_p
        writer.release()

    
    reader.release()

        
def gen_jsonl(json_path, augument_list):

    with open(json_path, 'r') as json_file:
        json_data = json_file.readlines()
        json_data = "[" + ",".join(json_data) + "]"
        json_data = json.loads(json_data)



    for augu in augument_list:

        s_p, e_p = augu

        base_dir = "/minerl/basalt-2022-behavioural-cloning-baseline/augumenteVideo"
        OUTPUT_FILE =  os.path.join(base_dir, json_path.split("/")[-1].split(".")[0] + f"_{s_p}_{e_p}.jsonl")

        with open(OUTPUT_FILE, 'w') as json_file:
            for i in range(s_p, e_p):
                json_line = json_data[i]
                json.dump(json_line, json_file)
                json_file.write("\n")




def augumentation(action, demo_tuples):

    MIN_INTERVAL = 60
    LEAST_LEN = 256
    LEAST_ACTION_NUM = 5

    total_t = 0

    for demo_tuple in tqdm(demo_tuples):
        video_path, json_path = demo_tuple

        agent_button_action_list = gen_agent_button_action_list(json_path)

        augument_list = []
        s_p = -1
        e_p = -1
        a_num = 0

        for i in range(len(agent_button_action_list)):
            agent_action, agent_button = agent_button_action_list[i]
            if action in agent_button:
                if s_p<0:
                    s_p = i
                    a_num += 1
                    continue
                elif e_p<0:
                    e_p = i
                    a_num += 1
                    continue
                else:
                    if i - e_p > MIN_INTERVAL:
                        if e_p - s_p > LEAST_LEN and a_num >= LEAST_ACTION_NUM:
                            augument_list.append([max(0, s_p-20), min(e_p+20, len(agent_button_action_list))])
                        s_p = i
                        e_p = -1
                        a_num = 1
                        
                    else:
                        e_p = i
                        a_num += 1

        total_t+=len(augument_list)
        # print(augument_list)
        # print(total_t)
        gen_video(video_path, augument_list)
        gen_jsonl(json_path, augument_list)




if __name__=="__main__":
    data_dir = "/data/MineRLBasaltBuildVillageHouse-v0"
    # video_path = os.path.join(data_dir, "cheeky-cornflower-setter-0a9ad3ddd136-20220726-193610.mp4")

    # json_path = os.path.join(data_dir, "cheeky-cornflower-setter-0a9ad3ddd136-20220726-193610.jsonl")

    # demo_tuple = [(video_path, json_path)]

    unique_ids = glob.glob(os.path.join(data_dir, "*.mp4"))
    unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))

    demo_tuple = []

    for uid in unique_ids:

        video_path = os.path.abspath(os.path.join(data_dir, uid + ".mp4"))
        json_path = os.path.abspath(os.path.join(data_dir, uid + ".jsonl"))
        demo_tuple.append((video_path, json_path))


    action = "use"


    augumentation(action, demo_tuple)