import cv2 as cv
import numpy as np
import random
from utils.tools import get_noop_action


SHOW = False

def do_none_ac(env, p=10, show=SHOW):
    ac = get_noop_action()

    for _ in range(p):
        obs, reward, done, info = env.step(ac)
        if done:
            raise ValueError('env done')
        if show:
            env.render()

    return obs

def get_cur_obs(env, p=10, show=SHOW):
    ac = get_noop_action()
    for _ in range(p):
        obs, reward, done, info = env.step(ac)
        if done:
            raise ValueError('env done')
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
        if done:
            raise ValueError('env done')
        if show:
            env.render()

def rate_random(data_list, rates_list):
    start =0
    random_num = random.random()
    for idx, score in enumerate(rates_list):
        start += score
        if random_num < start:
            break
    return data_list[idx]

def hotbar(env, idx, show=SHOW):
    ac = get_noop_action()
    ac[f'hotbar.{idx}'] = 1
    obs, reward, done, info = env.step(ac)
    if done:
        raise ValueError('env done')
    if show:
        env.render()
    do_none_ac(env,3)

def turn(env, towardstate, towards=[90,90], show=SHOW, min_diff=2, max_diff=10, return_water_percent=False):
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    p = [0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    total_dx = (towards[0] - towardstate.x + 360)%360
    total_dx = total_dx if abs(total_dx) <= 180 else total_dx-360
    total_dy = towards[1] - towardstate.y
    x_list = []
    y_list = []
    water_percent = []
    while abs(total_dx)>10 or abs(total_dy)>10:
        if abs(total_dx) > 10:
            cur_x = min(np.random.randint(min_diff,max_diff), abs(total_dx))
            if total_dx<0:
                cur_x = -cur_x
            total_dx -= cur_x
        else:
            cur_x = rate_random(x, p)
            if total_dx<0:
                cur_x = -cur_x
            total_dx -= cur_x
        if abs(total_dy) > 10:
            cur_y = min(np.random.randint(min_diff,max_diff), abs(total_dy))
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
        if done:
            raise ValueError('env done')
        if show: env.render()
        towardstate.mouse_move(x,y)
        do_none_ac(env,2)
        if return_water_percent:
            water_percent.append((get_blue_percent(obs['pov']), [towardstate.x, towardstate.y]))

    obs = turnOld(env, towardstate, towards)
    if return_water_percent:
        water_percent.append((get_blue_percent(obs['pov']), [towardstate.x, towardstate.y]))
    if return_water_percent:
        return obs, water_percent
    else:
        return obs

def turnOld(env, towardstate, towards=[90,90], show=SHOW):
    total_dx = towards[0] - towardstate.x
    total_dy = towards[1] - towardstate.y

    ac = get_noop_action()
    ac['camera'] = np.array([total_dy, total_dx])
    # ac['attack'] = 1
    obs, reward, done, info = env.step(ac)
    if done:
        raise ValueError('env done')
    if show:
        env.render()
    dy, dx = ac['camera']
    towardstate.mouse_move(dx,dy)
    obs = do_none_ac(env)
    return obs

def frame_diff(image1, image2, show=SHOW):
    difference = cv.subtract(image1, image2)
    # result = not np.any(difference) #if difference is all zeros it will return False
    # print(np.average(difference))
    return abs(np.average(difference))

def button_attack_1_block(env, show=SHOW):
    button(env,'attack',3)
    obs_1 = get_cur_obs(env, p=3)
    last_obs = obs_1
    for _ in range(250):
        ac = get_noop_action()
        ac['attack'] = 1
        obs, reward, done, info = env.step(ac)
        if done:
            raise ValueError('env done')
        if show:
            env.render()
        if frame_diff(obs_1['pov'], obs['pov']) > 5:
            print(frame_diff(obs_1['pov'], obs['pov']))
            print(frame_diff(last_obs['pov'], obs['pov']))
            break
        last_obs = obs
    do_none_ac(env, p=3)

def clearleaf(env,towardstate, show=SHOW):
    startx = towardstate.x
    starty = towardstate.y
    hotbar(env, 4)
    turn(env,towardstate, [startx,0])
    button_attack_1_block(env)
    # button(env, "attack",250)
    turn(env,towardstate, [startx,40])
    for _ in range(140):
        ac = get_noop_action()
        y = 0
        x = 3
        ac['camera'] = np.array([y,x])
        ac['attack'] = 1
        obs, reward, done, info = env.step(ac)
        if done:
            raise ValueError('env done')
        if show: env.render()
        towardstate.mouse_move(x,y)
    turn(env,towardstate, [startx,starty])
    hotbar(env, 2)

def buildpillar(env, height, towardstate, hb=3, show=SHOW):

    do_none_ac(env,3)

    startx = towardstate.x

    turn(env,towardstate,[towardstate.x,180])

    hotbar(env, hb)

    button(env, "attack", 1)
    
    do_none_ac(env,3)
    flag = True

    cur_h = 0 # 当前高度
    while cur_h < height:

        obs1 = get_cur_obs(env,3)['pov']

        button(env, 'jump', 3)

        do_none_ac(env, 1)

        button(env, 'use', 3)

        obs2 = get_cur_obs(env,3)['pov']


        if frame_diff(obs1,obs2) < 1 and flag:
            flag = False
            clearleaf(env,towardstate)
        else:
            cur_h += 1
            flag=True

    do_none_ac(env)

def backjump(env, p=4, show=SHOW):
    combine_button(env, ["back", "jump"], p)

def movejump(env, p=4,dir='back', show=SHOW):
    combine_button(env, [dir, "jump"], p)

def combine_button(env, a_list, p=5, show=SHOW):
    for _ in range(p):
        ac = get_noop_action()
        for a in a_list:
            ac[a] = 1
        obs, reward, done, info = env.step(ac)
        if done:
            raise ValueError('env done')
        if show:
            env.render()
    obs = do_none_ac(env,10)
    return obs



def makeWaterFall(env, towardstate, show=SHOW, high=3):
    turn(env,towardstate,[towardstate.x,180])
    # backjump(env)
    buildpillar(env,high,towardstate,2)
    do_none_ac(env)
    hotbar(env, 1)
    do_none_ac(env)
    button(env, 'use')
    do_none_ac(env, p=15)
    for _ in range(3):
        button(env, 'forward', 4)
        do_none_ac(env, p=9)
    do_none_ac(env, p=50)
    turn(env, towardstate, [towardstate.x - 180, 90], min_diff=15, max_diff=20)

    for _ in range(3):
        movejump(env, 20, dir='back')
    for _ in range(3):
        movejump(env, 10, dir='forward')


    turn(env,towardstate,[towardstate.x,30])
    turn(env, towardstate, [towardstate.x, 120])
    turn(env, towardstate, [towardstate.x, 65])

    movejump(env, 30, dir='left')
    movejump(env, 30, dir='right')

    water_percents = []
    _, water_percent = turn(env,towardstate,[towardstate.x - 25,towardstate.y], return_water_percent=True)
    water_percents.extend(water_percent)
    # turn(env,towardstate,[towardstate.x + 50,towardstate.y])
    _, water_percent = turn(env,towardstate,[towardstate.x + 180,towardstate.y], min_diff=15, max_diff=20, return_water_percent=True)
    water_percents.extend(water_percent)
    _, water_percent = turn(env,towardstate,[towardstate.x + 180,towardstate.y], min_diff=15, max_diff=20, return_water_percent=True)
    water_percents.extend(water_percent)
    _, water_percent = turn(env,towardstate,[towardstate.x + 25,towardstate.y], min_diff=15, max_diff=20, return_water_percent=True)
    water_percents.extend(water_percent)

    water_percents.sort(key=lambda x: x[0])
    turn(env, towardstate, water_percents[-1][1], min_diff=15, max_diff=20)
    do_none_ac(env, 50)
    button(env, 'ESC', 1)


def get_blue_percent(image):
    lower = np.array([100, 43, 46])
    upper = np.array([124, 255, 255])
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    bluemask = cv.inRange(hsv, lower, upper)[180:, :]
    blue_percent = np.mean(bluemask == 255)
    return blue_percent

def is_in_water(image):
    lower = np.array([100, 43, 46])
    upper = np.array([124, 255, 255])
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    bluemask = cv.inRange(hsv, lower, upper)[180:, :]
    blue_percent = np.mean(bluemask == 255)
    # cv2.imwrite('test1.png', bluemask)
    # print(blue_percent)
    if blue_percent > 0.5:
        return True
    else:
        return False

def near_magma(image):
    lower = np.array([8, 202, 100])
    upper = np.array([16, 255, 240])

    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    magma = cv.inRange(hsv, lower, upper)
    magma_percent = np.mean(magma == 255)
    # print(magma_percent)
    if magma_percent > 0.03:
        return True
    else:
        return False

def get_hsv_range(img):
    lows = []
    ups = []
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    for i in range(3):
        lows.append(hsv[:, :, i].min())
        ups.append(hsv[:, :, i].max())
    print(lows)
    print(ups)

def back(env,p=5, show=SHOW):
    button(env,'back',p=p)
    do_none_ac(env,5)

def forwardjump(env, p=4, show=SHOW):
    obs = combine_button(env, ["forward", "jump"], p)
    return obs

def is_stuck(env,towardstate,last_5_obs):
    for obs in last_5_obs[:-1]:
        if frame_diff(obs,last_5_obs[-1])<2:
            turn(env,towardstate, [towardstate.x,170])
            hotbar(env,4)
            button(env,'attack', 20)
            back(env,1)
            buildpillar(env,5,towardstate, 2)
            forwardjump(env)
            return
    return




if __name__ == "__main__":
    import cv2

    IMG_PATH = "/mnt/d/project/tmp/debug_sequence/443.png"
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # is_in_water(img)
    # is_lava_space(img)
    from utils.tools import BubbleCheck
    bubble_check = BubbleCheck('../utils/bubble_template.png', '../utils/bubble_mask.png')
    bubble_check.check(img)
