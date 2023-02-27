import gym
import minerl
import numpy as np
import time
import json
import random
import cv2 as cv
import os
from collections import OrderedDict

COUNTER_DIR = {
    "forward": "back",
    "back": "forward",
    "left": "right",
    "right": "left"
    }
SHOW = False

NOOP_ACTION = OrderedDict([('ESC', 0),
             ('attack', 0),
             ('back', 0),
             ('camera', np.array([0., 0.], dtype=np.float32)),
             ('drop', 0),
             ('forward', 0),
             ('hotbar.1', 0),
             ('hotbar.2', 0),
             ('hotbar.3', 0),
             ('hotbar.4', 0),
             ('hotbar.5', 0),
             ('hotbar.6', 0),
             ('hotbar.7', 0),
             ('hotbar.8', 0),
             ('hotbar.9', 0),
             ('inventory', 0),
             ('jump', 0),
             ('left', 0),
             ('pickItem', 0),
             ('right', 0),
             ('sneak', 0),
             ('sprint', 0),
             ('swapHands', 0),
             ('use', 0)])

def get_noop_action():
    return NOOP_ACTION.copy()


def do_none_ac(env, p=10, show=SHOW):
    ac = get_noop_action()

    for _ in range(p):
        obs, reward, done, info = env.step(ac)
        if show:
            env.render()

def frame_diff(image1, image2, show=SHOW):
    difference = cv.subtract(image1, image2)
    # result = not np.any(difference) #if difference is all zeros it will return False
    # print(np.average(difference))
    return abs(np.average(difference))

def get_cur_obs(env, p=10, show=SHOW):
    ac = get_noop_action()
    for _ in range(p):
        obs, reward, done, info = env.step(ac)
        if show:
            env.render()
    return obs

    

def button(env, a, p=3, show=SHOW):
    """
    env_action = {
        'ESC': 0, 
        'back': 0, 
        'drop': 0, 
        'forward': 0, 
        'hotbar.1': 0, 'hotbar.2': 0, 'hotbar.3': 0, 'hotbar.4': 0, 'hotbar.5': 0, 'hotbar.6': 0, 'hotbar.7': 0, 'hotbar.8': 0, 'hotbar.9': 0, 'inventory': 0, 
        'jump': 1, 
        'left': 0, 
        'right': 0, 
        'sneak': 0, 
        'sprint': 0, 
        'swapHands': 0, 
        'camera': array([-0.15,  0.  ]), 
        'attack': 0, 
        'use': 0, 
        'pickItem': 0
    }
    """
    for _ in range(p):
        ac = get_noop_action()
        ac[a] = 1
        obs, reward, done, info = env.step(ac)
        if show:
            env.render()

def button_attack_1_blockOld(env, show=SHOW):
    hotbar(env,1)
    button(env,'attack',3)
    obs_1 = get_cur_obs(env, p=3)
    last_obs = obs_1
    max_obs_diff = 1
    threshold = 2
    
    for i in range(1, 30):
        ac = get_noop_action()
        ac['attack'] = 1
        obs, reward, done, info = env.step(ac)
        if show:
            env.render()
        start_diff = frame_diff(obs_1['pov'], obs['pov'])
        cur_diff = frame_diff(last_obs['pov'], obs['pov'])
        if i == 1:
            print(f"start_diff: {start_diff}")
            if start_diff < 0.2: threshold = 2
            elif start_diff < 0.4: threshold = 2
        # print(f"max_obs_diff: {max_obs_diff}")
        # print(f"cur_diff: {cur_diff}")
        if start_diff > threshold:
            print(f"start_diff: {start_diff}")
            # if cur_diff > max_obs_diff*2:
            #     print(f"cur_diff: {cur_diff}")
            #     break
            break

        max_obs_diff = max(max_obs_diff, cur_diff)
        last_obs = obs
    obs_2 = get_cur_obs(env, p=3)
    if frame_diff(obs_1['pov'], obs_2['pov']) < 1:
        button_attack_1_blocknew(env)

def button_attack_1_block(env, show=SHOW):
    hotbar(env,1)

    obs_1 = get_cur_obs(env, p=3)
    last_obs = obs_1
    max_obs_diff = 1
    threshold = 2
    
    for i in range(25):
        ac = get_noop_action()
        ac['attack'] = 1
        obs, reward, done, info = env.step(ac)
        if show:
            env.render()
    obs_2 = get_cur_obs(env, p=3)
    if frame_diff(obs_1['pov'], obs_2['pov']) < 1:
        button_attack_1_blocknew(env)

def button_attack_1_blocknew(env, show=SHOW):
    att_list = [25,80,150]
    button(env,'attack',3)
    obs_1 = get_cur_obs(env)

    for att in att_list:
        for _ in range(att):
            ac = get_noop_action()
            ac['attack'] = 1
            obs, reward, done, info = env.step(ac)
            if show:
                env.render()
        obs_2 = get_cur_obs(env)
        diff = frame_diff(obs_1['pov'], obs_2['pov'])
        if diff > 2:
            break
    do_none_ac(env, p=3)


    # last_obs = obs_1
    # obs_list = [obs_1]
    # obs_diff_list = []
    # obs_d_diff_list = []

    # for i in range(10):
    #     ac = get_noop_action()
    #     ac['attack'] = 1
    #     obs, reward, done, info = env.step(ac)
    #     obs_list.append(obs)
    #     if show:
    #         env.render()
    #     obs_diff_list.append(frame_diff(obs['pov'], obs_1['pov']))
    #     last_obs = obs
    # max_diff = max(obs_diff_list)
    # per_diff = max_diff/100
    # print(f"per_diff: {per_diff}")
    # obs_2 = get_cur_obs(env)
    # print(frame_diff(obs_1['pov'], obs_2['pov']))
    # if frame_diff(obs_1['pov'], obs_2['pov'])> 3:
    #     return

    # for i in range(250):
    #     ac = get_noop_action()
    #     ac['attack'] = 1
    #     obs, reward, done, info = env.step(ac)
    #     obs_list.append(obs)
    #     if show:
    #         env.render()

    #     cur_diff = frame_diff(obs['pov'], obs_1['pov'])
    
    #     if cur_diff > per_diff*(i+10)*(i+10):
    #         print(f"cur_diff: {cur_diff}")
    #         break
    # do_none_ac(env, p=3)

def combine_button(env, a_list, p=5, show=SHOW):
    for _ in range(p):
        ac = get_noop_action()
        for a in a_list:
            ac[a] = 1
        obs, reward, done, info = env.step(ac)
        if show:
            env.render()
    do_none_ac(env,10)

def cal_rgb(img):
    R = np.mean(img[:,:,0])
    G = np.mean(img[:,:,1])
    B = np.mean(img[:,:,2])
    return R,G,B

def turn_horizontal(env,towardstate, show=SHOW):
    target_x_list = [0, 90, 180, 270]
    cur_x = towardstate.x
    cur_y = towardstate.y
    target_x_list_diff = [abs(cur_x-v) for v in target_x_list]
    target_x = np.array(target_x_list_diff).argmin()
    turnOld(env,towardstate,[target_x_list[target_x], 130])
    # ac = get_noop_action()
    # total_dx = target_x_list[target_x] - cur_x
    # total_dy = 90 - cur_y
    # ac['camera'] = np.array([total_dy, total_dx])
    # obs, reward, done, info = env.step(ac)
    # if show: env.render()
    # dy, dx = ac['camera']
    # towardstate.mouse_move(dx,dy)
    towardstate.x = 90
    img = get_cur_obs(env,5)['pov']
    R,G,B = cal_rgb(img)
    if (B > 80 and R<80 and G<80) or (B > 110 and R<90 and G<90) :
        return True, (R,G,B)
    img_right = np.array([v[500:] for v in img[:]] )
    img_left = np.array([v[:300] for v in img[:]] )
    img_top = np.array([v[:] for v in img[30:150]])
    img_bottom = np.array([v[:] for v in img[150:250]])
    for img_part in [img_right, img_left, img_top, img_bottom]:
        R,G,B = cal_rgb(img_part)
        if (B > 80 and R<80 and G<80) or (B >110 and R<90 and G<90) :
            return True, (R,G,B)
    return False, None

def forwardjump(env, p=4, show=SHOW):
    combine_button(env, ["forward", "jump"], p)

def backjump(env, p=4, show=SHOW):
    combine_button(env, ["back", "jump"], p)

def leftjump(env, p=5, show=SHOW):
    combine_button(env, ["left", "jump"], p)

def rightjump(env, p=5, show=SHOW):
    combine_button(env, ["right", "jump"], p)




def swaphands(env,towardstate, p=1, show=SHOW):
    # turn_horizontal(env,towardstate)
    # attack(env,10)
    button(env, "swapHands", 1)
    do_none_ac(env,15)
    button(env, "swapHands", 1)
    obs = get_cur_obs(env)
    do_none_ac(env,15)
    return obs

def forward(env,p=5, show=SHOW):
    button(env,'forward',p=p)
    do_none_ac(env,5)

def back(env,p=5, show=SHOW):
    button(env,'back',p=p)
    do_none_ac(env,5)

def backuse(env,p=5, show=SHOW):
    combine_button(env,['back','use'],p=p)
    do_none_ac(env,5)

def left(env,p=5, show=SHOW):
    button(env,'left',p=p)
    do_none_ac(env,5)

def right(env,p=5, show=SHOW):
    button(env,'right',p=p)
    do_none_ac(env,5)

def attack(env,p=3, show=SHOW):
    button(env,'attack',p=p)
    do_none_ac(env,5)

def use(env, p=1, show=SHOW):
    button(env, 'use', p)
    do_none_ac(env,5)

def hotbar(env, idx, show=SHOW):
    ac = get_noop_action()
    ac[f'hotbar.{idx}'] = 1
    obs, reward, done, info = env.step(ac)
    if show:
        env.render()
    do_none_ac(env,3)

def rate_random(data_list, rates_list):
    start =0
    random_num = random.random()
    for idx, score in enumerate(rates_list):
        start += score
        if random_num < start:
            break
    return data_list[idx]

def turn(env, towardstate, towards=[90,90], show=SHOW):
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    p = [0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    total_dx = (towards[0] - towardstate.x + 360)%360
    total_dx = total_dx if abs(total_dx) <= 180 else total_dx-360
    total_dy = towards[1] - towardstate.y
    x_list = []
    y_list = []
    while abs(total_dx)>10 or abs(total_dy)>10:
        if abs(total_dx) > 10:
            cur_x = min(np.random.randint(0,10), abs(total_dx))
            if total_dx<0:
                cur_x = -cur_x
            total_dx -= cur_x
        else:
            cur_x = rate_random(x, p)
            if total_dx<0:
                cur_x = -cur_x
            total_dx -= cur_x
        if abs(total_dy) > 10:
            cur_y = min(np.random.randint(0,10), abs(total_dy))
            if total_dy<0:
                cur_y = -cur_y
            total_dy -= cur_y
        else:
            cur_y = rate_random(x, p)
            if total_dy<0:
                cur_y = -cur_y
            total_dy -= cur_y
        x_list.append(cur_x)
        y_list.append(cur_y)
    x_list.append(total_dx)
    y_list.append(total_dy)

    for x,y in zip(x_list,y_list):
        ac = get_noop_action()
        ac['camera'] = np.array([y,x])
        obs, reward, done, info = env.step(ac)
        if show: env.render()
        towardstate.mouse_move(x,y)
        do_none_ac(env,2)
    turnOld(env, towardstate, towards)

def turnUse(env, towardstate, towards=[90,90], show=SHOW):
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    p = [0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    total_dx = (towards[0] - towardstate.x + 360)%360
    total_dx = total_dx if abs(total_dx) <= 180 else total_dx-360
    total_dy = towards[1] - towardstate.y
    x_list = []
    y_list = []
    while abs(total_dx)>10 or abs(total_dy)>10:
        if abs(total_dx) > 10:
            cur_x = min(np.random.randint(0,10), abs(total_dx))
            if total_dx<0:
                cur_x = -cur_x
            total_dx -= cur_x
        else:
            cur_x = rate_random(x, p)
            if total_dx<0:
                cur_x = -cur_x
            total_dx -= cur_x
        if abs(total_dy) > 10:
            cur_y = min(np.random.randint(0,10), abs(total_dy))
            if total_dy<0:
                cur_y = -cur_y
            total_dy -= cur_y
        else:
            cur_y = rate_random(x, p)
            if total_dy<0:
                cur_y = -cur_y
            total_dy -= cur_y
        x_list.append(cur_x)
        y_list.append(cur_y)
    x_list.append(total_dx)
    y_list.append(total_dy)

    for x,y in zip(x_list,y_list):
        ac = get_noop_action()
        ac['camera'] = np.array([y,x])
        ac['use'] = 1
        obs, reward, done, info = env.step(ac)
        if show: env.render()
        towardstate.mouse_move(x,y)
        do_none_ac(env,2)
    do_none_ac(env)

def turnOld(env, towardstate, towards=[90,90], show=SHOW):
    total_dx = towards[0] - towardstate.x
    total_dy = towards[1] - towardstate.y

    ac = get_noop_action()
    ac['camera'] = np.array([total_dy, total_dx])
    # ac['attack'] = 1
    obs, reward, done, info = env.step(ac)
    if show:
        env.render()
    dy, dx = ac['camera']
    towardstate.mouse_move(dx,dy)
    do_none_ac(env)

def turnattack(env, towardstate, towards=[90,90], show=SHOW):
    total_dx = towards[0] - towardstate.x
    total_dy = towards[1] - towardstate.y

    ac = get_noop_action()
    ac['camera'] = np.array([total_dy, total_dx])
    ac['attack'] = 1
    obs, reward, done, info = env.step(ac)
    if show:
        env.render()
    dy, dx = ac['camera']
    towardstate.mouse_move(dx,dy)

def inv_down(env, p=1.0, show=SHOW):
    ac = get_noop_action()
    ac['camera'] = np.array([p*2.5, 0])
    obs, reward, done, info = env.step(ac)
    if show:
        env.render()

def inv_up(env, p=1.0, show=SHOW):
    ac = get_noop_action()
    ac['camera'] = np.array([p*(-2.5), 0])
    obs, reward, done, info = env.step(ac)
    if show: env.render()

def inv_left(env, p=1.0, show=SHOW):
    ac = get_noop_action()
    ac['camera'] = np.array([0, p*(-2.5)])
    obs, reward, done, info = env.step(ac)
    if show: env.render()

def inv_right(env, p=1.0, show=SHOW):
    ac = get_noop_action()
    ac['camera'] = np.array([0, p*2.5])
    obs, reward, done, info = env.step(ac)
    if show: env.render()

def inventory(env, show=SHOW):
    button(env,'inventory',p=1)
    inv_down(env, 0.5)

def drop(env):
    do_none_ac(env,5)
    inv_right(env, 8)
    button(env, 'attack', 1)
    do_none_ac(env,5)
    inv_left(env, 8)

def inventory_toolbox(env, towardstate, show=SHOW):
    inventory(env)
    # 拿木块
    inv_down(env, 3.5)
    inv_left(env, 1)
    button(env,"attack",1)

    # crafting木板
    inv_right(env,2)
    inv_up(env, 7.5)
    button(env, "use", 1)
    do_none_ac(env,3)
    button(env, "use", 1)
    do_none_ac(env,3)
    button(env, "use", 1)
    do_none_ac(env,3)

    # 放木块
    inv_down(env, 6)
    button(env,"attack",1)
    drop(env)
    do_none_ac(env,3)
    
    # 拿木板
    inv_right(env, 3.5)
    inv_up(env, 5.5)
    button(env, "attack", 1)
    do_none_ac(env,3)
    button(env, "attack", 1)
    do_none_ac(env,3)
    button(env, "attack", 1)
    do_none_ac(env,3)
    button(env, "attack", 1)
    do_none_ac(env,3)

    # 造工具箱
    inv_left(env,3)
    button(env,"use",1)
    do_none_ac(env,3)
    inv_down(env,1)
    button(env,"use",1)
    do_none_ac(env,3)
    inv_right(env,1)
    button(env,"use",1)
    do_none_ac(env,3)
    inv_up(env,1)
    button(env,"use",1)
    do_none_ac(env,3)

    # 放木板
    inv_down(env, 5.5)
    button(env,"attack",1)
    drop(env)
    do_none_ac(env,3)

    # 拿工具箱
    inv_right(env, 2)
    inv_up(env, 5.5)
    button(env,"attack",1)
    do_none_ac(env,3)

    # 放工具箱
    inv_down(env, 7)
    inv_left(env, 5.5)
    button(env, "attack", 1)
    drop(env)
    do_none_ac(env, 3)

    # 关背包
    button(env, "inventory", 1)
    
    hotbar(env,4)
    turn(env, towardstate, [0,150])
    button(env,"use",1)
    turn(env, towardstate, [0,90])

def inventory_toolbox_small(env, towardstate, show=SHOW):
    inventory(env)
    # 拿木块
    inv_down(env, 3.5)
    inv_left(env, 1)
    button(env,"attack",1)

    # crafting木板
    inv_right(env,2)
    inv_up(env, 7.5)
    button(env, "use", 1)
    do_none_ac(env,3)
    button(env, "use", 1)
    do_none_ac(env,3)
    button(env, "use", 1)
    do_none_ac(env,3)

    # 放木块
    inv_down(env, 6)
    button(env,"attack",1)
    drop(env)
    do_none_ac(env,3)
    
    # 拿木板
    inv_right(env, 3.5)
    inv_up(env, 5.5)
    button(env, "attack", 1)
    do_none_ac(env,3)
    button(env, "attack", 1)
    do_none_ac(env,3)
    button(env, "attack", 1)
    do_none_ac(env,3)
    button(env, "attack", 1)
    do_none_ac(env,3)

    # 造工具箱
    inv_left(env,3)
    button(env,"use",1)
    do_none_ac(env,3)
    inv_down(env,1)
    button(env,"use",1)
    do_none_ac(env,3)
    inv_right(env,1)
    button(env,"use",1)
    do_none_ac(env,3)
    inv_up(env,1)
    button(env,"use",1)
    do_none_ac(env,3)

    # 放木板
    inv_down(env, 5.5)
    button(env,"attack",1)
    drop(env)
    do_none_ac(env,3)

    # 拿工具箱
    inv_right(env, 2)
    inv_up(env, 5.5)
    button(env,"attack",1)
    do_none_ac(env,3)

    # 放工具箱
    inv_down(env, 7)
    inv_left(env, 5.5)
    button(env, "attack", 1)
    drop(env)
    do_none_ac(env, 3)

    # 关背包
    button(env, "inventory", 1)
    
    hotbar(env,4)

def inventory_door(env,towardstate, show=SHOW):
    turn(env, towardstate, [0,150])
    button(env,"use",1)
    do_none_ac(env)
    inv_down(env, 2.5)
    inv_right(env, 2)
    button(env, "attack", 1)
    inv_left(env,5)
    inv_up(env,6)
    button(env,"use",1)
    do_none_ac(env, 5)

    inv_down(env,1)
    button(env,"use",1)
    do_none_ac(env, 5)

    inv_down(env,1)
    button(env,"use",1)
    do_none_ac(env, 5)

    inv_up(env,2)
    inv_right(env,1)

    button(env,"use",1)
    do_none_ac(env, 5)
    inv_down(env,1)
    button(env,"use",1)
    do_none_ac(env, 5)
    inv_down(env,1)
    button(env,"use",1)
    do_none_ac(env, 5)

    inv_right(env,2)
    inv_down(env,4)
    button(env,"attack",1)
    do_none_ac(env,5)
    drop(env)
    inv_right(env,3)
    inv_up(env,5)
    button(env,"attack",1)
    do_none_ac(env,5)
    inv_down(env,6.5)
    inv_left(env,4)
    button(env,"attack",1)
    do_none_ac(env,5)
    button(env, "inventory", 1) 
    hotbar(env,4)
    turn(env, towardstate, [90,90])

def build_door(env,towardstate, show=SHOW):
        
    turn(env,towardstate,[90,130])
    use(env)

    turn(env,towardstate,[50,110])
    use(env)
    
    turn(env,towardstate,[60,90])
    use(env)

    turn(env,towardstate,[90,90])
    use(env)

    forward(env)
    left(env)
    forward(env,20)
    turn(env, towardstate, [270, 80])


def clearleaf(env,towardstate, show=SHOW):
    startx = towardstate.x
    starty = towardstate.y
    hotbar(env, 1)
    turn(env,towardstate, [startx,0])
    button_attack_1_block(env)
    # button(env, "attack",250)
    turnattack(env,towardstate, [startx,40])
    for _ in range(140):
        ac = get_noop_action()
        y = 0
        x = 3
        ac['camera'] = np.array([y,x])
        ac['attack'] = 1
        obs, reward, done, info = env.step(ac)
        if show: env.render()
        towardstate.mouse_move(x,y)
    turn(env,towardstate, [startx,starty])
    hotbar(env, 3)



def buildpillar(env, height, towardstate, hb=3, show=SHOW):

    do_none_ac(env,3)

    # for _ in range(10):
    #     ac = get_noop_action()
    #     ac['camera'] = np.array([10, 0])
    #     obs, reward, done, info = env.step(ac)
    #     if show: env.render()
    #     dy, dx = ac['camera']
    #     towardstate.mouse_move(dx,dy)

    startx = towardstate.x

    # clearleaf(env,towardstate)

    turn(env,towardstate,[towardstate.x,180])

    hotbar(env, hb)

    button(env, "attack", 1)
    obs_1 = get_cur_obs(env,5)
    # do_none_ac(env,3)
    flag = True

    cur_h = 0 # 当前高度
    while cur_h < height:
        
        # do_none_ac(env,3)
        obs1 = get_cur_obs(env,3)['pov']

        button(env, 'jump', 3)

        do_none_ac(env, 1)

        button(env, 'use', 3)

        obs2 = get_cur_obs(env,3)['pov']


        if frame_diff(obs1,obs2) < 1 and flag:
            flag = False
            clearleaf(env,towardstate)
            # turn(env, towardstate, [random.choice([0,90,180,270]), 10])
            # button(env, "attack",10)
            # turn(env, towardstate, [startx, 180])
        else:
            cur_h += 1
            flag=True


    # for _ in range(10):
    #     ac = get_noop_action()
    #     ac['camera'] = np.array([-10, 0])
    #     obs, reward, done, info = env.step(ac)
    #     if show: env.render()
    #     dy, dx = ac['camera']
    #     towardstate.mouse_move(dx,dy)

    obs_2 = get_cur_obs(env,5)
    if frame_diff(obs_1['pov'],obs_2['pov']) < 1:
        button_attack_1_block(env)
        buildpillar(env, height+1, towardstate, hb, show)


def flatten_check(env, direction,towardstate, show=SHOW):
    obs_1 = get_cur_obs(env)

    eval(direction)(env)
    eval(direction)(env)
    obs_2 = get_cur_obs(env)
    eval(COUNTER_DIR[direction])(env)
    eval(COUNTER_DIR[direction])(env)
    obs_3 = get_cur_obs(env)
    
    obs_diff = frame_diff(obs_1['pov'], obs_3['pov'])
    # print(f"obs_diff: {obs_diff}")
    if obs_diff < 3:
        return True, None
    else:
        eval(direction)(env)
        obs_4 = get_cur_obs(env)
        obs_diff = frame_diff(obs_2['pov'], obs_4['pov'])
        # print(f"obs_diff: {obs_diff}")
        if obs_diff < 3:
            return False, "high"
        else:
            return False, "low" 

def flatten(env, direction, towardstate, show=SHOW):
    ret, diff = flatten_check(env, direction,towardstate)
    if ret:
        eval(direction)(env)
        return 0
    else:
        # 检测的方向矮了，已经掉下去了，填地就行, 水平面保持不变
        if diff == "low":
            eval(COUNTER_DIR[direction])(env,5)
            buildpillar(env, 1, towardstate, 8)
            return 0

        # 检测的方向高了，先填地，再移动，水平面+1
        elif diff == 'high':
            eval(direction)(env,5)
            buildpillar(env, 1, towardstate, 8)
            eval(direction)
            return 1

def flatten_areaOld(env, towardstate, show=SHOW, width=4):
    cur_pos = [0, 0]
    area_height = [[-1 for _ in range(width)] for _ in range(width)]
    buildpillar(env, 1, towardstate, 8)
    area_height[cur_pos[0]][cur_pos[1]] = 1
    # 逐行检测平地，铺平
    for y in range(width):
        if y != 0:
            height = flatten(env, "forward", towardstate)
            forward(env,1)
            if height==1:
                area_height[0][y-1] = 1
                area_height[0][y] = 1
            else:
                area_height[0][y] = area_height[0][y-1]
        for x in range(width-1):
            height = flatten(env, "right", towardstate)
            if height==1:
                area_height[x][y] = 1
                area_height[x+1][y] = 1
            else:
                area_height[x+1][y] = area_height[x][y]

        for x in range(width-1)[::-1]:
            left(env)
            if area_height[x+1][y] == area_height[x][y]:
                continue
            else:
                buildpillar(env, 1, towardstate, 8)
                area_height[x][y] = area_height[x+1][y]

        right(env,3)


def check_flatten_area(env, towardstate,show=SHOW):
    obs1=get_cur_obs(env)
    back(env,13)
    right(env,13)
    forward(env,13)
    left(env,13)
    obs2=get_cur_obs(env)
    return frame_diff(obs1['pov'],obs2['pov'])<3

def flatten_area(env, towardstate, show=SHOW, width=4):
    
    for yi in range(4):

        buildpillar(env, 1, towardstate, 8)
        for _ in range(3):
            right(env,10)
            left(env, 10)
            combine_button(env, ['right','attack'],2)
            button(env, 'left', 3)
            buildpillar(env, 1, towardstate, 8)
        left(env)
        left(env)
        left(env)
        right(env,1)
        if yi!=3:
            forward(env,10)
            back(env,10)
            forward(env,1)
            combine_button(env, ['right','attack'],2)
            button(env, 'left', 2)

    return check_flatten_area(env, towardstate)






def jumpuse(env, show=SHOW):
    button(env, 'jump')
    do_none_ac(env,p=1)
    button(env, 'use')
    do_none_ac(env)




def makeWaterFall(env, towardstate, show=SHOW):
    turn(env,towardstate,[towardstate.x,180])
    # backjump(env)
    buildpillar(env,10,towardstate,2)
    do_none_ac(env)
    hotbar(env, 1)
    do_none_ac(env)
    button(env, 'use')
    do_none_ac(env)
    button(env, 'back',10)
    turn(env, towardstate, [towardstate.x,90])

    for _ in range(3):
        backjump(env,5)

    turn(env,towardstate,[towardstate.x,50])

    do_none_ac(env, 50)




def build_4_pillar(env, towardstate, show=SHOW):
    buildpillar(env, 3, towardstate)

    back(env)
    back(env)
    forward(env)
    forward(env)
    buildpillar(env, 1, towardstate)
    back(env)
    buildpillar(env, 1, towardstate)
    back(env)

    buildpillar(env, 3, towardstate)

    forward(env, 1)
    right(env)
    right(env)
    left(env)
    left(env)
    buildpillar(env, 1, towardstate)
    right(env)
    buildpillar(env, 1, towardstate)
    right(env)

    buildpillar(env, 3, towardstate)

    forward(env)
    forward(env)
    back(env)
    back(env)
    buildpillar(env, 1, towardstate)
    forward(env)
    buildpillar(env, 1, towardstate)
    forward(env)

    buildpillar(env, 3, towardstate)

def clearforwardOld(env,towardstate, show=SHOW):
    startx = towardstate.x
    starty = towardstate.y
    hotbar(env, 2)
    turn(env,towardstate, [startx-30,90])
    for _ in range(30):
        ac = get_noop_action()
        y = 0
        x = 2
        ac['camera'] = np.array([y,x])
        ac['attack'] = 1
        obs, reward, done, info = env.step(ac)
        if show: env.render()
        towardstate.mouse_move(x,y)
    turn(env,towardstate, [startx,starty])
    hotbar(env, 3)

def cleardown(env,towardstate, show=SHOW):
    startx = towardstate.x
    starty = towardstate.y
    hotbar(env, 2)
    turn(env,towardstate, [startx-30,150])
    for _ in range(30):
        ac = get_noop_action()
        y = 0
        x = 2
        ac['camera'] = np.array([y,x])
        ac['attack'] = 1
        obs, reward, done, info = env.step(ac)
        if show: env.render()
        towardstate.mouse_move(x,y)
    turn(env,towardstate, [startx,starty])
    hotbar(env, 3)

def build_4_pillarV2(env, towardstate, show=SHOW):
    buildpillar(env, 3, towardstate)

    turn(env,towardstate, [270,towardstate.y])
    # clearforward(env, towardstate)
    # cleardown(env, towardstate)
    forward(env)
    forward(env)
    back(env)
    back(env)

    buildpillar(env, 1, towardstate)
    # clearforward(env, towardstate)
    forward(env)
    buildpillar(env, 1, towardstate)
    # clearforward(env, towardstate)
    forward(env)

    buildpillar(env, 3, towardstate)
    turn(env,towardstate, [180,towardstate.y])

    left(env, 1)

    # clearforward(env, towardstate)
    # cleardown(env, towardstate)
    forward(env)
    forward(env)
    back(env)
    back(env)

    buildpillar(env, 1, towardstate)
    # clearforward(env, towardstate)
    forward(env)
    buildpillar(env, 1, towardstate)
    # clearforward(env, towardstate)
    forward(env)

    buildpillar(env, 3, towardstate)
    turn(env,towardstate, [90,towardstate.y])

    # clearforward(env, towardstate)
    # cleardown(env, towardstate)
    forward(env)
    forward(env)
    back(env)
    back(env)

    buildpillar(env, 1, towardstate)
    # clearforward(env, towardstate)
    forward(env)
    buildpillar(env, 1, towardstate)
    # clearforward(env, towardstate)
    forward(env)

    buildpillar(env, 3, towardstate)

def build_bottom_beam(env, towardstate, show=SHOW):

    beam_pos = [
        [120, 120],[30, 120],[0, 130],[340, 120]
    ]
    left(env,10)
    back(env,10)
    right(env,10)
    do_none_ac(env)


    # for pos in beam_pos:
    #     turn(env, towardstate, pos)
    #     button(env, "use", 1)

def build_top_beam(env, towardstate, show=SHOW):

    beam_pos = [
        [120, 60],[130, 50],[98, 65],[50, 70],[40, 70],[15, 70],[340, 70],[320, 50]
    ]

    for pos in beam_pos:
        turn(env, towardstate, pos)
        button(env, "use", 1)

def build_window(env, towardstate, show=SHOW):
    beam_pos = [
        [120, 90],[200, 90],[240, 90],[330, 90],[350, 90],[40, 90]
    ]

    hotbar(env,5)
    for pos in beam_pos:
        turn(env, towardstate, pos)
        button(env, "use", 1)

def build_roof(env, towardstate, show=SHOW):

    beam_pos = [
        [40, 70],[60, 60],[70, 50],[0, 70]
    ]

    hotbar(env,3)
    for pos in beam_pos:
        turn(env, towardstate, pos)
        button(env, "use", 1)  

    turn(env,towardstate, [0,90])

def look_around(env, towardstate, show=SHOW):

    dx = 3
    dy = 0

    for i in range(120):
        ac = get_noop_action()
        ac['camera'] = np.array([dy,dx])
        obs, reward, done, info = env.step(ac)
        if show: env.render()
        towardstate.mouse_move(dx,dy)
    do_none_ac(env)

def look_around_pics(env, towardstate, show=SHOW):

    dx = 3
    dy = 0
    pics = []
    for i in range(120):
        ac = get_noop_action()
        ac['camera'] = np.array([dy,dx])
        obs, reward, done, info = env.step(ac)
        if show: env.render()
        towardstate.mouse_move(dx,dy)
        if i%30==29:
            pics.append(obs['pov'])
    do_none_ac(env)
    return pics


def go_circle_old(env, towardstate, show=SHOW):
    dir_list = ['right','forward','left','back']
    turn(env,towardstate, [90,130])
    obs_1 = get_cur_obs(env)['pov']
    last_obs = obs_1
    for dir in dir_list:
        for _ in range(4):
            eval(dir)(env)
            cur_obs = get_cur_obs(env,1)['pov']
            # print(frame_diff(last_obs,cur_obs))
            if frame_diff(last_obs,cur_obs)<3:
                return False
            last_obs = cur_obs
    right(env)
    forward(env)
    for dir in dir_list:
        for _ in range(2):
            eval(dir)(env)
            cur_obs = get_cur_obs(env,1)['pov']
            # print(frame_diff(last_obs,cur_obs))
            if frame_diff(last_obs,cur_obs)<3:
                return False
            last_obs = cur_obs
    right(env)
    forward(env)
    back(env)
    back(env)
    left(env)
    left(env)

    # right(env,20)
    # do_none_ac(env)
    # forward(env,20)
    # do_none_ac(env)
    # left(env,20)
    # do_none_ac(env)
    # back(env,20)
    do_none_ac(env)
    obs_2 = get_cur_obs(env)['pov']
    obs_diff = frame_diff(obs_1,obs_2)
    # print(obs_diff)
    if obs_diff<3:
        return True
    return False


def go_circle_jump(env, towardstate, show=SHOW):
    dir_list = ['rightjump','forwardjump','leftjump','backjump']
    turn(env,towardstate, [90,130])
    obs_1 = get_cur_obs(env)['pov']
    last_obs = obs_1
    for dir in dir_list:
        for _ in range(5):
            eval(dir)(env)
            cur_obs = get_cur_obs(env,1)['pov']
            # print(frame_diff(last_obs,cur_obs))
            if frame_diff(last_obs,cur_obs)<3:
                return False
            last_obs = cur_obs

    # right(env,20)
    # do_none_ac(env)
    # forward(env,20)
    # do_none_ac(env)
    # left(env,20)
    # do_none_ac(env)
    # back(env,20)
    do_none_ac(env)
    obs_2 = get_cur_obs(env)['pov']
    obs_diff = frame_diff(obs_1,obs_2)
    # print(obs_diff)
    if obs_diff<3:
        return True
    flatten_area(env,towardstate)
    return False

def buildVillageHouse(env, towardstate, show=SHOW):

    do_none_ac(env)

    if not flatten_area(env,towardstate):
        return False

    build_4_pillarV2(env,towardstate)

    build_bottom_beam(env, towardstate)

    build_top_beam(env,towardstate)

    build_window(env, towardstate)

    build_roof(env, towardstate)

    inventory_toolbox(env, towardstate)

    inventory_door(env, towardstate)

    build_door(env,towardstate)

    return True


def flattenV2(env, towwardstate):
    obs1 = get_cur_obs(env)
    right(env)
    obs2 = get_cur_obs(env)
    left(env)
    obs3 = get_cur_obs(env)
    if frame_diff(obs1,obs3)<3:
        right(env)
        return
    else:
        left(env)
        obs4 = get_cur_obs(env)
        if frame_diff(obs4,obs3)<3:
            buildpillar(env,1,towwardstate, 8)
        else:
            buildpillar(env,1,towwardstate, 8)

def flatten_areaV2(env, towardstate):
    buildpillar(env, 1, towardstate, 8)
    cur_pos = [0,0]


    obs_1 = get_cur_obs(env)
    right(env)
    obs_2 = get_cur_obs(env)
    left(env)
    obs_3 = get_cur_obs(env)

    if obs_1==obs_3:
        right(env)
        cur_pos=[1,0]


def is_stuck(env,towardstate,last_obs):
    for obs in last_obs[:-1]:
        if frame_diff(obs,last_obs[-1])<2:
            turn(env,towardstate, [towardstate.x,170])
            hotbar(env,1)
            button(env,'attack', 20)
            back(env,1)
            buildpillar(env,5,towardstate)
            forwardjump(env)
            return
    return

def is_inv(img):
    blood = np.array([v[230:300] for v in img[320:330]])
    r,g,b = cal_rgb(blood)
    if r<50 and g<30 and b<30:
        # plt.imshow(pic)
        # plt.show()
        return True
    return False

def forwarduse(env, towardstate, p=5, show=SHOW):
    turnOld(env, towardstate, [towardstate.x,170])
    button(env,'attack',2)
    do_none_ac(env,2)
    for _ in range(p):
        ac = get_noop_action()
        ac['forward'] = 1
        ac['use'] = 1
        obs, reward, done, info = env.step(ac)
        if show: env.render()
        if is_inv(obs['pov']):
            button(env,'inventory',1)
    do_none_ac(env,5)

def forwarduseV2(env, towardstate, p=20, show=SHOW):
    turnOld(env, towardstate, [towardstate.x,170])
    button(env,'attack',2)
    do_none_ac(env,2)
    for _ in range(p):
        ac = get_noop_action()
        ac['forward'] = 1
        ac['use'] = 1
        obs, reward, done, info = env.step(ac)
        if show: env.render()
        ac = get_noop_action()
        ac['use'] = 1
        obs, reward, done, info = env.step(ac)
        if show: env.render()
    do_none_ac(env,10)

def go_circle_2_check(env,towardstate, show=SHOW):

    hotbar(env,8)
    for i in range(10):
        forwarduse(env, towardstate, 20)
        button(env,'use',2)
        do_none_ac(env)
        back(env,20)
        do_none_ac(env)
        right(env,2)

    left(env,19)
    do_none_ac(env)

def right_2(env):
    right(env,2)
    do_none_ac(env,2)

def go_circle(env, towardstate, show=SHOW):
    dir = [90,175]
    cur_dir = dir
    dis_list = [4,4,4,3,3,2,2,1,2,1,3,3]
    turn_horizontal(env,towardstate)
    obs1 = get_cur_obs(env)['pov']
    last_obs = obs1
    hotbar(env,7)
    flag = True
    for dis in dis_list:
        turnOld(env, towardstate, cur_dir)
        step = 0
        while step<dis:
            forwarduse(env, towardstate, show=SHOW)
            cur_obs = get_cur_obs(env,1)['pov']
            if frame_diff(last_obs,cur_obs)<1:
                forwardjump(env)
                obs3 = get_cur_obs(env)['pov']
                if frame_diff(obs3,cur_obs)<3:
                    return False
                else:
                    back(env,3)
                
                buildpillar(env,1,towardstate,7)
                hotbar(env,7)
                turnOld(env, towardstate, cur_dir)
                if flag:
                    flag = False
                    step -= 2
                else:
                    return False
            step+=1
            last_obs = cur_obs
        button(env,'use',1)
        cur_dir[0] = (cur_dir[0] + 90)%360
        button(env,'use',1)
    if flag==False:
        for dis in dis_list:
            turnOld(env, towardstate, cur_dir)
            step = 0
            while step<dis:
                forwarduse(env, towardstate, show=SHOW)
                cur_obs = get_cur_obs(env,1)['pov']
                if frame_diff(last_obs,cur_obs)<1:
                    return False
                step+=1
                last_obs = cur_obs
            cur_dir[0] = (cur_dir[0] + 90)%360
    # back(env,5)
    # back(env,5)
    # left(env,5)
    # left(env,5)
    turnOld(env, towardstate, [90, 130])
    obs2 = get_cur_obs(env)['pov']
    do_none_ac(env)
    # print(frame_diff(obs1,obs2))
    if frame_diff(obs1,obs2)<3:
        return True
    else:
        go_circle_2_check(env,towardstate)
    return go_circle_old(env, towardstate, show=SHOW)




def change_material_sand(env, towardstate, show=SHOW):
    inventory(env)
    inv_left(env)
    button(env,'attack',1)
    inv_down(env,3.5)
    inv_left(env)
    button(env,'attack',1)
    do_none_ac(env,1)
    inv_up(env,3.5)
    inv_right(env)
    button(env,'attack',1)
    do_none_ac(env,1)

    inv_right(env)
    button(env,'attack',1)
    do_none_ac(env,1)
    inv_down(env,3.5)
    inv_right(env,2)
    button(env,'attack',1)
    do_none_ac(env,1)
    inv_left(env,2)
    inv_up(env,3.5)
    button(env,'attack',1)
    do_none_ac(env,1)

    inv_right(env)
    button(env,'attack',1)
    do_none_ac(env,1)
    inv_down(env,3.5)
    inv_right(env,2)
    button(env,'attack',1)
    do_none_ac(env,1)
    inv_left(env,2)
    inv_up(env,3.5)
    button(env,'attack',1)
    do_none_ac(env,1)

    inv_down(env,3.5)
    inv_right(env)
    button(env,'attack',1)
    do_none_ac(env,1)
    inv_left(env,4)
    button(env,'attack',1)
    do_none_ac(env,1)
    inv_right(env,4)
    button(env,'attack',1)
    do_none_ac(env,1)


    inventory(env)


def change_material_snow(env, towardstate, show=SHOW):
    inventory(env)
    inv_right(env,4)
    button(env,'attack',1)
    inv_down(env,3.5)
    inv_left(env,6)
    button(env,'attack',1)
    do_none_ac(env,1)
    inv_up(env,3.5)
    inv_right(env,6)
    button(env,'attack',1)
    do_none_ac(env,1)

    inv_down(env,3.5)
    inv_left(env)
    button(env,'attack',1)
    do_none_ac(env,1)
    inv_up(env,2.5)
    inv_left(env,6)
    button(env,'attack',1)
    do_none_ac(env,1)
    inv_right(env,6)
    inv_down(env,2.5)
    button(env,'attack',1)
    do_none_ac(env,1)

    inventory(env)



def lowturnuse(env, towardstate, h=1, show=SHOW):
    corner_dirlist = [[45,150], [155,150], [225,145],[300,145]]
    # vertex_dirlist = [[360,155], [155,150], [225,145],[300,145]]
    hotbar(env,8)
    for _ in range(h):
        buildpillar(env,1,towardstate)
        turnOld(env, towardstate, [300,160])
        for corner in corner_dirlist:
            turnOld(env, towardstate, corner)
            use(env)
        # for _ in range(20):
        #     dx = 20
        #     dy = np.random.randint(-5,5)
        #     ac = get_noop_action()
        #     ac['attack'] = 1
        #     ac['camera'] = np.array([dy,dx])
        #     obs, reward, done, info = env.step(ac)
        #     if show: env.render()
        #     towardstate.mouse_move(dx,dy)

        # for _ in range(20):
        #     dx = 20
        #     dy = np.random.randint(-5,5)
        #     ac = get_noop_action()
        #     ac['use'] = 1
        #     ac['camera'] = np.array([dy,dx])
        #     obs, reward, done, info = env.step(ac)
        #     if show: env.render()
        #     towardstate.mouse_move(dx,dy)



    
def zhongjibaodi(env, towardstate, show=SHOW):
    buildpillar(env,1,towardstate)
    back(env)
    back(env)
    forward(env)
    forward(env)
    left(env)
    buildpillar(env,1,towardstate)
    right(env)
    right(env)
    left(env)
    left(env)
    lowturnuse(env, towardstate, show=show)
    

    
def process_theta(thetas):
    
    thetapos = [v for v in thetas if v > 5]
    thetaneg = [v for v in thetas if v < -5]
    if all([v%90<5 for v in thetas]):
        print("okkk")
        return 0
    if len(thetapos)>0:
        return -thetapos[0]
    else:
        return -thetaneg[0]   
 
import math
def cal_ang(x1,x2,y1,y2):
    angle = math.atan2(y2-y1,x2-x1)
    return angle* (180/np.pi)

def get_theta(img):
    # 灰度处理 & 双边滤波 & 边缘检测
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray_blur = cv.bilateralFilter(img_gray, 7, 25, 25)
    edges = cv.Canny(img_gray_blur, 50, 150)

    # 霍夫变换
    lines = cv.HoughLinesP(edges[50:300, 150:500], 1, np.pi/180,60) 
    
    # print(f"lines: {lines}")

    # # 坐标提取
    thetas = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            thetas.append(cal_ang(x1,x2,y1,y2))
    # print(f"thetas: {thetas}")
    return thetas

def hough_recover(env,towardstate,show=SHOW):
    turnOld(env, towardstate, [towardstate.x, 180])
    attack(env,3)
    obs = get_cur_obs(env, 5)

    total_theta = 0
    while total_theta<360*2:
        

        thetas = get_theta(obs['pov'])
        if total_theta ==0:
            if all([v%90<10 for v in thetas]):
                towardstate.x = 90
                print("okkk")
                return 0
        if all([v%90<1 for v in thetas]):
            towardstate.x = 90
            print("okkk")
            return 0
        ac = get_noop_action()
        ac['camera'] = np.array([0, 1])
        obs, reward, done, info = env.step(ac)
        if show:
            env.render()
        total_theta += 1


def go_circle_2_check_small(env,towardstate, show=SHOW):

    hotbar(env,7)
    for i in range(6):
        forwarduse(env, towardstate, 10)
        
        button(env,'use',2)
        do_none_ac(env)
        backuse(env,10)
        do_none_ac(env)
        if i!=5: right(env,2)

    left(env,11)
    do_none_ac(env)

def go_circle_old_small(env, towardstate, show=SHOW):
    dir_list = ['right','forward','left','back']
    turn(env,towardstate, [90,130])
    obs_1 = get_cur_obs(env)['pov']
    last_obs = obs_1
    for dir in dir_list:
        for _ in range(2):
            eval(dir)(env)
            cur_obs = get_cur_obs(env,1)['pov']
            # print(frame_diff(last_obs,cur_obs))
            if frame_diff(last_obs,cur_obs)<3:
                return False
            last_obs = cur_obs
    obs_2 = get_cur_obs(env)['pov']
    obs_diff = frame_diff(obs_1,obs_2)
    # print(obs_diff)
    if obs_diff<3:
        return True
    return False

def go_circle_small(env, towardstate, show=SHOW):
    dir = [90,175]
    cur_dir = dir
    dis_list = [2,2,2,2]
    turn_horizontal(env,towardstate)
    obs1 = get_cur_obs(env)['pov']
    last_obs = obs1
    hotbar(env,7)
    flag = True
    for dis in dis_list:
        turn(env, towardstate, cur_dir)
        
        img = get_cur_obs(env,1)['pov']
        R,G,B = cal_rgb(img)
        if (B > 80 and R<80 and G<80) or (B > 110 and R<90 and G<90) :
            return False

        step = 0
        while step<dis:
            forwarduse(env, towardstate, show=SHOW)
            use(env,3)
            cur_obs = get_cur_obs(env,1)['pov']
            if frame_diff(last_obs,cur_obs)<1:
                forwardjump(env)
                obs3 = get_cur_obs(env)['pov']
                if frame_diff(obs3,cur_obs)<3:
                    return False
                else:
                    back(env,3)
                
                buildpillar(env,1,towardstate,7)
                hotbar(env,7)
                turn(env, towardstate, cur_dir)
                if flag:
                    flag = False
                    step -= 2
                else:
                    return False
            step+=1
            last_obs = cur_obs
        button(env,'use',1)
        cur_dir[0] = (cur_dir[0] + 90)%360
        button(env,'use',1)
    # if flag==False:
    #     for dis in dis_list:
    #         turnOld(env, towardstate, cur_dir)
    #         step = 0
    #         while step<dis:
    #             forwarduse(env, towardstate, show=SHOW)
    #             use(env,3)
    #             cur_obs = get_cur_obs(env,1)['pov']
    #             if frame_diff(last_obs,cur_obs)<1:
    #                 return False
    #             step+=1
    #             last_obs = cur_obs
    #         cur_dir[0] = (cur_dir[0] + 90)%360
    # back(env,5)
    # back(env,5)
    # left(env,5)
    # left(env,5)
    turn(env, towardstate, [90, 130])
    obs2 = get_cur_obs(env)['pov']
    do_none_ac(env)
    # print(frame_diff(obs1,obs2))
    if frame_diff(obs1,obs2)<1:
        return True
    else:
        go_circle_2_check_small(env,towardstate)
    return go_circle_old_small(env, towardstate, show=SHOW)

def clearforward(env,towardstate,show=SHOW):
    forward(env,2)
    do_none_ac(env,5)
    # turnattack(env,towardstate,[towardstate.x, 150])
    # turnattack(env,towardstate,[towardstate.x, 180])
    obs_1 = get_cur_obs(env,1)['pov']
    forward(env)
    obs_2 = get_cur_obs(env,1)['pov']
    if frame_diff(obs_1,obs_2)<3:
        hotbar(env,1)
        turn(env,towardstate,[towardstate.x,150])
        # button(env,'attack',50)
        button_attack_1_block(env)
        forward(env)
        hotbar(env,3)
    back(env,10)

def clearforwardV2(env,towardstate,show=SHOW):
    forward(env,2)
    do_none_ac(env,5)
    # turnattack(env,towardstate,[towardstate.x, 150])
    # turnattack(env,towardstate,[towardstate.x, 180])
    obs_1 = get_cur_obs(env,1)['pov']
    forward(env)
    obs_2 = get_cur_obs(env,1)['pov']
    if frame_diff(obs_1,obs_2)<3:
        hotbar(env,1)
        turn(env,towardstate,[towardstate.x,150])
        button(env,'attack',50)
        forward(env)
        hotbar(env,3)
    button(env,'attack',7)
    do_none_ac(env)
    button(env,'attack',7)
    back(env,10) 

def baodi(env, towardstate, show=SHOW):
    if not go_circle_small(env, towardstate):
        return False, False
    right(env,6)
    forward(env,6)
    hotbar(env,2)
    turn(env,towardstate, [90,180])
    # button(env,'attack',30)
    button_attack_1_block(env)

    button(env,'back',2)
    obs_1 = get_cur_obs(env)['pov']
    do_none_ac(env,2)
    button(env,'right',2)
    do_none_ac(env,2)
    button(env,'forward',4)
    do_none_ac(env,4)
    button(env,'left',4)
    do_none_ac(env,4)
    button(env,'back',4)
    do_none_ac(env,4)
    button(env,'right',2)
    do_none_ac(env,2)

    # combine_button(env,['left','back'],7)
    # combine_button(env,['forward','right'],7)
    # combine_button(env,['forward','right'],7)
    # combine_button(env,['left','back'],7)

    # combine_button(env,['right','back'],7)
    # combine_button(env,['forward','left'],7)
    # combine_button(env,['forward','left'],7)
    # combine_button(env,['right','back'],7)
    obs_2 = get_cur_obs(env)['pov']

    if frame_diff(obs_1,obs_2)<5:
        obs_1 = get_cur_obs(env)['pov']
        combine_button(env,['forward','left'],7)
        if frame_diff(obs_1,obs_2)>5:
            return False,False

    combine_button(env,['forward','left'],7)

    # button(env,'left',5)
    # button(env,'right',5)
    # button(env,'right',5)
    # button(env,'left',5)
    # button(env,'forward',5)
    # button(env,'back',5)
    # button(env,'back',5)
    # button(env,'forward',5)
    # button(env,'forward',5)
    # button(env,'left',5)
    
    turn(env,towardstate,[90,160])
    button(env,'attack',30)
    hotbar(env,8)
    use(env,1)
    # button_attack_1_block(env)
    
    buildpillar(env,1,towardstate,3)
    
    # turn(env,towardstate,[90,160])
    # hotbar(env,8)
    # use(env,1)
    hough_recover(env, towardstate)
    left(env,4)
    forward(env,4)
    buildpillar(env,3,towardstate)
    turn(env,towardstate,[270,180])
    clearforwardV2(env,towardstate)


    buildpillar(env,1,towardstate)
    clearforward(env,towardstate)

    buildpillar(env,3,towardstate)
    turn(env,towardstate,[180,180])
    clearforwardV2(env,towardstate)

    buildpillar(env,1,towardstate)
    clearforward(env,towardstate)

    buildpillar(env,3,towardstate)
    turn(env,towardstate,[90,180])
    clearforwardV2(env,towardstate)

    buildpillar(env,1,towardstate)
    clearforward(env,towardstate)
    buildpillar(env,3,towardstate)

    # forward(env,10)
    # buildpillar(env,1,towardstate)

    # back(env)
    # forward(env,7)
    # buildpillar(env,3,towardstate)

    # right(env,7)
    # left(env,10)
    # buildpillar(env,1,towardstate)

    # right(env)
    # left(env,7)
    # buildpillar(env,3,towardstate)


    # forward(env,7)
    # back(env,10)
    # buildpillar(env,1,towardstate)

    # forward(env)
    # back(env,7)
    # buildpillar(env,3,towardstate)

    left(env,6)
    back(env,6)
    back(env)
    right(env)

    inventory_toolbox_small(env,towardstate)
    turn(env,towardstate,[90,180])
    jumpuse(env)

    # door
    use(env,1)
    do_none_ac(env)
    inv_down(env, 2.5)
    inv_right(env, 2)
    button(env, "attack", 1)
    inv_left(env,5)
    inv_up(env,6)
    button(env,"use",1)
    do_none_ac(env, 5)

    inv_down(env,1)
    button(env,"use",1)
    do_none_ac(env, 5)

    inv_down(env,1)
    button(env,"use",1)
    do_none_ac(env, 5)

    inv_up(env,2)
    inv_right(env,1)

    button(env,"use",1)
    do_none_ac(env, 5)
    inv_down(env,1)
    button(env,"use",1)
    do_none_ac(env, 5)
    inv_down(env,1)
    button(env,"use",1)
    do_none_ac(env, 5)

    inv_right(env,2)
    inv_down(env,4)
    button(env,"attack",1)
    do_none_ac(env,5)
    drop(env)
    inv_right(env,3)
    inv_up(env,5)
    button(env,"attack",1)
    do_none_ac(env,5)
    inv_down(env,6.5)
    inv_left(env,4)
    button(env,"attack",1)
    do_none_ac(env,5)
    button(env, "inventory", 1) 
    hotbar(env,4)
    turn(env, towardstate, [90,90])
    # end door
    turn(env, towardstate, [90,150])
    use(env,1)
    hotbar(env,2)
    turn(env, towardstate, [90,180])
    button(env,'attack',20)
    combine_button(env,['right','back'],2)
    do_none_ac(env, 5)

    # check ok
    obs_inv_1 = get_cur_obs(env)['pov']
    combine_button(env,['right','back'])
    obs_inv_2 = get_cur_obs(env)['pov']
    if frame_diff(obs_inv_1,obs_inv_2)>5:
        print(f"checkdiff{frame_diff(obs_inv_1,obs_inv_2)}")
        return False, True
    combine_button(env,['left','forward'],10)
    obs_inv_1 = get_cur_obs(env)['pov']
    combine_button(env,['left','forward'])
    obs_inv_2 = get_cur_obs(env)['pov']
    if frame_diff(obs_inv_1,obs_inv_2)>5:
        return False, True
    combine_button(env,['right','back'],10)
    # end check

    hotbar(env,3)
    

    turn(env,towardstate,[105,50])
    use(env,5)
    turn(env,towardstate,[135,60])
    use(env,1)
    turn(env,towardstate,[30,70])
    use(env,1)
    turn(env,towardstate,[305,60])
    use(env,1)

    # window
    hotbar(env,5)
    turn(env,towardstate,[310,90])
    use(env,1)
    turn(env,towardstate,[340,90])
    use(env,1)
    turn(env,towardstate,[220,90])
    use(env,1)

    turn(env,towardstate,[90,90])
    use(env,1)
    combine_button(env,['forward','left'],15)


    turn(env,towardstate,[270,90])
    backjump(env)
    backjump(env)
    backjump(env)

    return True, True


def cal_style_val(val_list):
    res = []
    for val in val_list:
        r_avg = np.mean([v[0] for v in val])
        g_avg = np.mean([v[1] for v in val])
        b_avg = np.mean([v[2] for v in val])
        res.append([r_avg, g_avg, b_avg])
    return res

def cal_pics_avg(pics):
    all = []
    right = []
    left = []
    up = []
    bottom = []
    for pic in pics:
        img = pic
        img_right = np.array([v[500:] for v in img[:]] )
        img_left = np.array([v[:300] for v in img[:]] )
        img_top = np.array([v[:] for v in img[30:150]])
        img_bottom = np.array([v[:] for v in img[150:250]])

        all.append(list(cal_rgb(img)))
        right.append(list(cal_rgb(img_right)))
        left.append(list(cal_rgb(img_left)))
        up.append(list(cal_rgb(img_top)))
        bottom.append(list(cal_rgb(img_bottom)))
    return cal_style_val([all,right,left,up,bottom])

def check_style(env, towardstate, show=SHOW):
    style_val_dict = {'snow': [[127.16135406332671, 129.83498739252647, 135.77869977678571],
                                [102.63624196900983, 102.37499102418745, 103.53676681783827],
                                [130.9732885802469, 133.71350352733685, 139.27396075837746],
                                [117.63328993055555, 125.3981618923611, 147.77602244543652],
                                [114.24513653273809, 111.56451450892857, 104.45706770833333]],
                      'sand': [[132.41417859331233, 138.70119558239966, 134.69022587680905],
                                [113.17632402961185, 116.94938358676805, 114.14922735760972],
                                [137.24162426081543, 143.8820527544351, 139.25357158418922],
                                [135.98465204831933, 155.89614408263301, 178.02751433385856],
                                [124.49387815126052, 120.67339285714284, 99.34859742647059]],
                       'jungle': [[93.86806033721751, 99.52995446386534, 90.22881885593219],
                                [87.161505582459, 89.57426251008879, 82.57613902340597],
                                [94.1028730382925, 100.49795182046454, 90.39299152542374],
                                [104.31121998587571, 116.96808847545906, 130.7695291534251],
                                [83.69535248940679, 83.46646371822033, 59.66217703919491]]}
    pics = look_around_pics(env, towardstate)

    cur_val = cal_pics_avg(pics)
    styles = ['snow', 'sand', 'jungle']
    diff = []
    for k,v in style_val_dict.items():
        diff.append(frame_diff(np.array(cur_val), np.array(v)))
    
    style = styles[np.argmin(diff)]
    print(f"{style} is the best style")
    if style =='snow':
        change_material_sand(env, towardstate)
    elif style =='sand':
        change_material_sand(env, towardstate)


    
def baodiV2(env, towardstate,show=SHOW):
    if not go_circle_small(env, towardstate):
        return False,True
    right(env,6)
    forward(env,6)
    hotbar(env,2)
    turn(env,towardstate, [90,180])
    # button(env,'attack',30)
    button_attack_1_block(env)

    button(env,'back',2)
    obs_1 = get_cur_obs(env)['pov']
    do_none_ac(env,2)
    button(env,'right',2)
    do_none_ac(env,2)
    button(env,'forward',4)
    do_none_ac(env,4)
    button(env,'left',4)
    do_none_ac(env,4)
    button(env,'back',4)
    do_none_ac(env,4)
    button(env,'right',2)
    do_none_ac(env,2)

    # combine_button(env,['left','back'],7)
    # combine_button(env,['forward','right'],7)
    # combine_button(env,['forward','right'],7)
    # combine_button(env,['left','back'],7)

    # combine_button(env,['right','back'],7)
    # combine_button(env,['forward','left'],7)
    # combine_button(env,['forward','left'],7)
    # combine_button(env,['right','back'],7)
    obs_2 = get_cur_obs(env)['pov']

    if frame_diff(obs_1,obs_2)<5:
        obs_1 = get_cur_obs(env)['pov']
        combine_button(env,['forward','left'],7)
        if frame_diff(obs_1,obs_2)>5:
            return False,True

    combine_button(env,['forward','left'],7)

    # button(env,'left',5)
    # button(env,'right',5)
    # button(env,'right',5)
    # button(env,'left',5)
    # button(env,'forward',5)
    # button(env,'back',5)
    # button(env,'back',5)
    # button(env,'forward',5)
    # button(env,'forward',5)
    # button(env,'left',5)
    
    turn(env,towardstate,[90,160])
    button(env,'attack',30)
    hotbar(env,8)
    use(env,1)
    # button_attack_1_block(env)
    
    buildpillar(env,1,towardstate,3)
    
    # turn(env,towardstate,[90,160])
    # hotbar(env,8)
    # use(env,1)
    hough_recover(env, towardstate)
    left(env,4)
    forward(env,4)
    buildpillar(env,3,towardstate)
    turn(env,towardstate,[270,180])
    clearforward(env,towardstate)


    buildpillar(env,1,towardstate)
    clearforward(env,towardstate)

    buildpillar(env,3,towardstate)
    turn(env,towardstate,[180,180])
    clearforward(env,towardstate)

    buildpillar(env,1,towardstate)
    clearforward(env,towardstate)

    buildpillar(env,3,towardstate)
    turn(env,towardstate,[90,180])
    clearforward(env,towardstate)

    buildpillar(env,1,towardstate)
    clearforward(env,towardstate)
    buildpillar(env,3,towardstate)

    left(env,6)
    back(env,6)
    back(env)
    right(env)

    buildpillar(env,1,towardstate,7)

    hotbar(env,4)
    turn(env, towardstate, [90,150])
    use(env,1)
    hotbar(env,2)
    turn(env, towardstate, [90,180])
    button(env,'attack',20)
    
    # check ok
    obs_inv_1 = get_cur_obs(env)['pov']
    combine_button(env,['right','back'])
    obs_inv_2 = get_cur_obs(env,3)['pov']
    if frame_diff(obs_inv_1,obs_inv_2)>3:
        return False, True
    combine_button(env,['left','forward'],10)
    obs_inv_1 = get_cur_obs(env,5)['pov']
    combine_button(env,['left','forward'])
    obs_inv_2 = get_cur_obs(env,3)['pov']
    if frame_diff(obs_inv_1,obs_inv_2)>3:
        return False, True
    combine_button(env,['right','back'],10)
    # end check

    hotbar(env,3)
    

    turn(env,towardstate,[105,50])
    use(env,5)
    turn(env,towardstate,[135,60])
    use(env,1)
    turn(env,towardstate,[30,70])
    use(env,1)
    turn(env,towardstate,[300,60])
    use(env,1)

    # window
    hotbar(env,5)
    turn(env,towardstate,[310,90])
    use(env,1)
    turn(env,towardstate,[340,90])
    use(env,1)
    turn(env,towardstate,[220,90])
    use(env,1)

    turn(env,towardstate,[90,90])
    use(env,1)
    combine_button(env,['forward','left'],15)


    turn(env,towardstate,[270,90])
    backjump(env)
    backjump(env)
    backjump(env)

    return True,True
