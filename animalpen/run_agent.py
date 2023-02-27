import os
import aicrowd_gym
import pickle
import random
from argparse import ArgumentParser

from animalpen.openai_vpt.agent import MineRLAgent
from animalpen.animal_detector import AnimalDetector
from animalpen.script import Handler


def main(model, weights, env, n_episodes=3, max_steps=int(6000), show=False):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    VPT_MODEL = os.path.join(base_dir, 'train', 'animal_pen_foundation-model-1x.model')
    VPT_WEIGHT = os.path.join(base_dir, 'train', 'animal_pen_MineRLBasalt.weights')
    ANIMAL_DETECTOR_WEIGHT = os.path.join(base_dir, 'train', 'animal_detector.pt')
    ANIMAL_DETECTOR_FILE = os.path.join(base_dir, 'train', 'animal_detector.yaml')

    # unzip file
    # os.system(f"""cd {base_dir}/hub && 7z x myfiles.zip.001 -y""")
    # os.system(f"""mv {base_dir}/hub/checkpoints/animal_pen_foundation-model-1x.model {base_dir}/train/""")
    # os.system(f"""mv {base_dir}/hub/checkpoints/animal_pen_MineRLBasalt.weights {base_dir}/train/""")
    # os.system(f"""mv {base_dir}/hub/checkpoints/animal_detector.pt {base_dir}/train/""")
    # os.system(f"""mv {base_dir}/hub/checkpoints/animal_detector.yaml {base_dir}/train/""")
    # os.system(f"""mv {base_dir}/hub/checkpoints/animal_pen_flattenareaV4resnet50.pt {base_dir}/train/""")
    # os.system(f"""mv {base_dir}/hub/checkpoints/animal_pen_block_1.png {base_dir}/train/""")
    # os.system(f"""mv {base_dir}/hub/checkpoints/animal_pen_block_2.png {base_dir}/train/""")

    # state machine
    STATUS_PRE_RANDOM_COLLECT = -2
    STATUS_RANDOM_COLLECT = -1
    STATUS_RANDOM_WALK = 0
    STATUS_PURSUE = 1
    STATUS_PURSUE_WAIT = 2
    STATUS_RANDOM_WALK_WITH_ANIMAL = 3
    STATUS_PLANE_RECHECK = 4
    STATUS_RESET_AGENT_ANGLE_POS = 5
    STATUS_PLANE_FILL = 6
    STATUS_RESET_AGENT_ANGLE_POS_2 = 7
    STATUS_ANIMAL_PEN = 8
    STATUS_MAKE_HOLE = 9
    STATUS_PRE_PLANE_FILL = 10
    STATUS_RESET_AGENT_ANGLE_POS2 = 11
    STATUS_PRE_PLANE_FILL_RABBIT = 12
    STATUS_ANIMAL_PEN_RABBIT = 13
    STATUS_RANDOM_COLLECT_CHECK = 14
    STATUS_PRE_PLANE_FILL_BACK = 15
    STATUS_STUCK = 16
    STATUS_RESET_AGENT_ANGLE_POS_2_RABBIT = 17
    STATUS_MINIMUM_GUARANTEE = 98
    STATUS_MINIMUM_GUARANTEE_WALK = 99
    STATUS_FINAL = 100

    # env
    env = aicrowd_gym.make(env)

    # bc agent
    agent_parameters = pickle.load(open(VPT_MODEL, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(VPT_WEIGHT)

    for _ in range(n_episodes):
        # yolo
        animal_detector = AnimalDetector(ANIMAL_DETECTOR_WEIGHT, ANIMAL_DETECTOR_FILE)

        # handler
        handler = Handler(env, base_dir)

        # run
        status_random_walk_with_animal_count = 0
        status_random_walk_with_animal_count2 = 0
        status_random_walk_with_animal_count_max = random.randint(100, 120)

        MAX_STUCK_NUM = 2

        obs = env.reset()
        handler.reset(obs['pov'])

        pre_plane_kill_count = 0
        random_walk_count = 0
        has_stuck = 0
        status = STATUS_PRE_RANDOM_COLLECT
        for index in range(max_steps):
            # ====================================
            #           yolo detect
            # ====================================
            yolo_res = animal_detector.inference(obs['pov'])
            handled_img, handled_yolo_res = handler.cv_handle(obs['pov'], yolo_res)

            # ====================================
            #          state mechine
            # ====================================
            if status == STATUS_PRE_RANDOM_COLLECT:
                """
                find a place for collecting wood
                """
                action = agent.get_action(obs)
                action['camera'] = action['camera'][0]
                action = handler.action_mask(action)
                action = handler.action_mask_with_water(action, obs['pov'])

                is_plane = handler.is_plane_space(obs['pov'])
                if is_plane:
                    status = STATUS_RANDOM_COLLECT
                    print('change to STATUS_RANDOM_COLLECT')

            elif status == STATUS_RANDOM_COLLECT:
                """
                random walk to collect items
                - via vpt model
                - via yolo detect result
                """
                action, is_collect = handler.collect_item()
                if is_collect:
                    status = STATUS_RANDOM_COLLECT_CHECK
                    print('change to STATUS_RANDOM_COLLECT_CHECK')

            elif status == STATUS_RANDOM_COLLECT_CHECK:
                """
                random walk to collect items
                """
                action, is_collect = handler.collect_item_check(obs['pov'])
                if is_collect:
                    status = STATUS_RANDOM_WALK
                    print('change to STATUS_RANDOM_WALK')

            elif status == STATUS_RANDOM_WALK:
                """
                random walk
                - via vpt model
                - via yolo detect result
                """
                random_walk_count += 1
                action = agent.get_action(obs)
                action['camera'] = action['camera'][0]
                action = handler.action_mask_with_yolo(action, handled_yolo_res)
                action = handler.action_mask_in_water(action, obs['pov'])
                # action = handler.action_mask_with_water_limit(action, obs['pov'])

                if handler.detect_animal_flag:
                    status = STATUS_PURSUE
                    print('change to STATUS_PURSUE')

                # 脱困脚本
                if random_walk_count % 100 == 0 and has_stuck < MAX_STUCK_NUM:
                    is_stuck = handler.is_stuck()
                    if is_stuck:
                        has_stuck += 1
                        status = STATUS_STUCK
                        print('change to STATUS_STUCK')

                if index > 5500: # 快结束了，还没找到动物，先建篱笆
                    status = STATUS_MINIMUM_GUARANTEE
                    handler.record_detect_animal = 'rabbit'
                    print('change to STATUS_MINIMUM_GUARANTEE')
                
                # 脱困，可操作加一
                if index > 2500 and MAX_STUCK_NUM == 2:
                    MAX_STUCK_NUM += 1
                if index > 3500 and MAX_STUCK_NUM == 3:
                    MAX_STUCK_NUM += 1
                if index > 4500 and MAX_STUCK_NUM == 4:
                    MAX_STUCK_NUM += 1

            elif status == STATUS_STUCK:
                action, done = handler.stuck_action()
                if done:
                    status = STATUS_RANDOM_WALK
                    print('change to STATUS_RANDOM_WALK')

            elif status == STATUS_PURSUE:
                """
                get close to animal via script action 
                    (if not detect animal, keep random walk)
                """
                action = agent.get_action(obs)
                action['camera'] = action['camera'][0]
                action = handler.action_mask(action)
                action, is_pursue = handler.pursue_action(action, handled_yolo_res)

                if is_pursue:
                    status = STATUS_PURSUE_WAIT
                    print('change to STATUS_PURSUE_WAIT')

            elif status == STATUS_PURSUE_WAIT:
                """
                wait for animal to be in front of agent
                """
                action, is_pursue_and_wait = handler.wait()

                if is_pursue_and_wait:
                    status = STATUS_RANDOM_WALK_WITH_ANIMAL
                    print('change to STATUS_RANDOM_WALK_WITH_ANIMAL')

            elif status == STATUS_RANDOM_WALK_WITH_ANIMAL:
                """
                random walk with animal for finding plant
                """
                action = agent.get_action(obs)
                action['camera'] = action['camera'][0]
                action = handler.action_mask(action)
                action = handler.action_mask_for_attract_animal(action)
                action = handler.action_mask_with_water(action, obs['pov'])
                action = handler.action_mask_in_water(action, obs['pov'])

                status_random_walk_with_animal_count += 1
                status_random_walk_with_animal_count2 += 1
                if status_random_walk_with_animal_count == status_random_walk_with_animal_count_max:
                    status_random_walk_with_animal_count -= 1

                    action, is_success_wait_animal = handler.load_back_wait_animal(yolo_res)
                    if is_success_wait_animal:
                        status_random_walk_with_animal_count = 0
                        status_random_walk_with_animal_count_max = random.randint(200, 300)
                        print('success wait animal')
                
                if status_random_walk_with_animal_count2 > 30:
                    is_plane = handler.is_plane_space(obs['pov']) and handler.is_plane_space_via_plane_detect_model(obs['pov'])
                    if is_plane:
                        status = STATUS_PLANE_RECHECK
                        print('change to STATUS_PLANE_RECHECK')

            elif status == STATUS_PLANE_RECHECK:
                """
                recheck if the plane is still there
                """
                action, is_success_recheck = handler.recheck_plane_space_water_limit(obs['pov'])
                if is_success_recheck == True:
                    status = STATUS_RESET_AGENT_ANGLE_POS
                    print('change to STATUS_RESET_AGENT_ANGLE_POS')
                elif is_success_recheck == False:
                    status = STATUS_RANDOM_WALK_WITH_ANIMAL
                    print('fail recheck, change to STATUS_RANDOM_WALK_WITH_ANIMAL')

            elif status == STATUS_RESET_AGENT_ANGLE_POS:
                """
                reset agent angle position
                """
                action, is_reset, reset_angle_count = handler.reset_agent_pos_angle_action_v2(obs['pov'])
                if is_reset:
                    status = STATUS_RESET_AGENT_ANGLE_POS2
                    handler.reset_agent_pos_angle_action_v2_flag = 0
                    print('change to STATUS_RESET_AGENT_ANGLE_POS2')

                if handler.is_water_space(obs['pov']):
                    status = STATUS_RANDOM_WALK_WITH_ANIMAL
                    print('detect water, change to STATUS_RANDOM_WALK_WITH_ANIMAL')

            elif status == STATUS_RESET_AGENT_ANGLE_POS2:
                """
                reset agent angle position
                """
                action, is_reset = handler.reset_agent_pos_angle_action_v2_1(obs['pov'])
                if is_reset:
                    if handler.record_detect_animal == 'rabbit':
                        status = STATUS_PRE_PLANE_FILL_RABBIT
                        print('change to STATUS_PRE_PLANE_FILL_RABBIT')
                    else:
                        status = STATUS_PRE_PLANE_FILL
                        print('change to STATUS_PRE_PLANE_FILL')
            
            # ==============================
            #  pig sheep cow handler
            # ==============================
            elif status == STATUS_PRE_PLANE_FILL:
                """
                pre plane fill
                """
                action, is_pre_plane_fill = handler.plane_detect_via_fill_v5_pre_v2(obs['pov'])
                pre_plane_kill_count += 1
                if is_pre_plane_fill:
                    status = STATUS_MAKE_HOLE
                    print('change to STATUS_MAKE_HOLE')
                
                if pre_plane_kill_count > 2000 or handler.is_in_water(obs['pov']):
                    status = STATUS_PRE_PLANE_FILL_BACK
                    handler.plane_detect_via_fill_status = 0
                    handler.plane_detect_via_fill_index = 0 # we add this line to avoid the abnormal error when the porgram running in current status. This will cause the subsequent episodes cannot be tested.
                    print('change to STATUS_PRE_PLANE_FILL_BACK')

            elif status == STATUS_PRE_PLANE_FILL_BACK:
                """
                pre plane fill back
                """
                action, is_pre_plane_fill_back = handler.plane_detect_via_fill_v5_pre_back(obs['pov'])
                if is_pre_plane_fill_back:
                    status = STATUS_RANDOM_WALK_WITH_ANIMAL
                    print('change to STATUS_RANDOM_WALK_WITH_ANIMAL')

            elif status == STATUS_MAKE_HOLE:
                action, done = handler.make_hole_v3(obs['pov'])
                if done:
                    status = STATUS_PLANE_FILL
                    print('change to STATUS_PLANE_FILL')

                if handler.is_in_water(obs['pov']):
                    status = STATUS_PRE_PLANE_FILL_BACK
                    print('change to STATUS_PRE_PLANE_FILL_BACK')

            elif status == STATUS_PLANE_FILL:
                """
                fill the plane
                """
                action, is_fill = handler.plane_detect_via_fill_v5(obs['pov'])
                if is_fill:
                    status = STATUS_RESET_AGENT_ANGLE_POS_2
                    print('change to STATUS_RESET_AGENT_ANGLE_POS_2')

            elif status == STATUS_RESET_AGENT_ANGLE_POS_2:
                action, is_reset = handler.reset_agent_pos_angle_action_v2_2(obs['pov'])
                if is_reset:
                    status = STATUS_ANIMAL_PEN

            elif status == STATUS_ANIMAL_PEN:
                """
                create animal pen
                """
                action, is_pen, current_pen_index, total_pen_index = handler.pen_script()
                print('start pen... [{}/{}]'.format(current_pen_index, total_pen_index))
                
                if is_pen: # save video
                    status = STATUS_FINAL

            # =============================
            #  rabbit handler
            # =============================
            elif status == STATUS_PRE_PLANE_FILL_RABBIT:
                action, is_fill = handler.plane_detect_via_fill_v2_2(obs['pov'])
                if is_fill:
                    status = STATUS_RESET_AGENT_ANGLE_POS_2_RABBIT
                    print('change to STATUS_RESET_AGENT_ANGLE_POS_2_RABBIT')

            elif status == STATUS_RESET_AGENT_ANGLE_POS_2_RABBIT:
                action, is_reset, _ = handler.reset_agent_pos_angle_action_1(obs['pov'])
                if is_reset:
                    status = STATUS_ANIMAL_PEN_RABBIT

            elif status == STATUS_ANIMAL_PEN_RABBIT:
                action, is_pen, current_pen_index, total_pen_index = handler.pen_script_rabbit()
                print('start pen... [{}/{}]'.format(current_pen_index, total_pen_index))
                
                if is_pen: # save video
                    status = STATUS_FINAL

            # =============================
            #  minimum guarantee
            # =============================
            elif status == STATUS_MINIMUM_GUARANTEE:
                action, done = handler.minimum_guarantee()
                if done:
                    status = STATUS_MINIMUM_GUARANTEE_WALK

            elif status == STATUS_MINIMUM_GUARANTEE_WALK:
                action = agent.get_action(obs)
                action['camera'] = action['camera'][0]
                action = handler.action_mask_with_yolo(action, handled_yolo_res)

            # ====================================
            # env step
            handler.step(action, obs['pov'])    # script
            action['ESC'] = 0
            if status == STATUS_FINAL:
                action['ESC'] = 1
            obs, _, done, _ = env.step(action)

            if show:
                env.render()
            if done:
                print("reset env...")
                break

    env.close()



if __name__ == '__main__':
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--env", type=str, default='MineRLBasaltMakeWaterfall-v0')
    parser.add_argument("--show", action="store_true", help="Render the environment.")

    args = parser.parse_args()

    main(args.model, args.weights, args.env, show=args.show, n_episodes=10)