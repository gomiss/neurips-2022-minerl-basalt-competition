import cv2
import numpy as np


class OptialFlow():
    def __init__(self) -> None:
        self.feature_params = {
            'maxCorners': 300,
            'qualityLevel': 0.1,
            'minDistance': 5,
            'blockSize': 3
        }
        self.lk_params = {
            'winSize': (15, 15),
            'maxLevel': 2,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
        self.color = np.random.randint(0, 255, (300, 3))
        self.frame_top = 30
        self.frame_bottom = 320

        self.old_frame = None
        self.old_frame_gray = None
        self.p0 = None

        self.max_direction_list = 5
        self.direction_list = []

    def reset(self, frame):
        self.old_frame = frame.copy()[self.frame_top:self.frame_bottom, ...]
        self.old_frame_gray = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY)
        self.old_frame_gray[230:290, 400:640] = 0 # mask agent tool
        self.p0 = cv2.goodFeaturesToTrack(self.old_frame_gray, mask = None, **self.feature_params)
        self.mask = np.zeros_like(self.old_frame)
    
    def step(self, frame):
        frame = frame.copy()[self.frame_top:self.frame_bottom, ...]
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray[230:290, 400:640] = 0 # mask agent tool
        
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.old_frame_gray, 
            frame_gray, 
            self.p0, 
            None, 
            **self.lk_params)
        
        # Select good points
        good_new = None
        if p1 is not None:
            good_new = p1[st==1]
            good_old = self.p0[st==1]
        
        # draw the tracks
        good_diff = []
        ori_corner_poses = []
        if good_new is not None:
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()

                # add diff
                diff = np.array([a - c, b - d])
                if np.linalg.norm(diff) >= 1.:
                    good_diff.append(diff / np.linalg.norm(diff))
                    ori_corner_poses.append(np.array([c, d]))

                self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
                frame = cv2.circle(frame.astype(np.uint8), (int(a), int(b)), 5, self.color[i].tolist(), -1)
            
        img = cv2.add(frame, self.mask)

        # Now update the previous frame and previous points
        self.old_frame_gray = frame_gray.copy()
        if good_new is not None:
            self.p0 = good_new.reshape(-1, 1, 2)

        self.mask = np.zeros_like(self.old_frame)
        self.p0 = cv2.goodFeaturesToTrack(self.old_frame_gray, mask = None, **self.feature_params)
        return img[...,::-1], np.array(good_diff), np.array(ori_corner_poses)

    def handle_direction(self, good_diffs, ori_corner_poses, show=False):
        DIR_DICT = {
            0: 'left',
            1: 'up',
            2: 'right',
            3: 'down'
        }
        MIDDLE_PIXEL = 320
        SECTOR_FORWARD_BACKWARD_THRESHOLD = 2
        SECTOR_FORWARD_BACKWARD_THRESHOLD2 = 10
        N_GOOD_DIFF_THRESHOLD = 10

        # need enough good_diffs to handle
        if len(good_diffs) < N_GOOD_DIFF_THRESHOLD:
            return None

        # ==============================================
        # get diff section 
        sector = {idx: 0 for idx in range(4)}
        for diff in good_diffs:
            angle = np.arctan2(diff[1], diff[0])
            if angle < 0:
                angle += 2 * np.pi
            
            if angle < np.pi / 4 or angle > 7 * np.pi / 4:
                sector[0] += 1
            elif angle > np.pi / 4 and angle < 3 * np.pi / 4:
                sector[1] += 1
            elif angle > 3 * np.pi / 4 and angle < 5 * np.pi / 4:
                sector[2] += 1
            elif angle > 5 * np.pi / 4 and angle < 7 * np.pi / 4:
                sector[3] += 1
    
        # ==============================================
        # handle four diection (left right up down)
        n_good_diff = len(good_diffs)
        useful_sector = None
        for key in sector:
            if sector[key] > n_good_diff // 2:
                useful_sector = DIR_DICT[key]

        # ==============================================
        # handle forward backward 
        #   (it will override the four direction)
        sector_count = 0
        for key in sector:
            if sector[key] > SECTOR_FORWARD_BACKWARD_THRESHOLD:
                sector_count += 1
        
        if sector[0] > SECTOR_FORWARD_BACKWARD_THRESHOLD2 and \
           sector[2] > SECTOR_FORWARD_BACKWARD_THRESHOLD2 and \
           sector_count > 2:
            
            # get diff convergence & dispersion
            convergence_count = 0
            dispersion_count = 0
            for diff, ori_corner_pos in zip(good_diffs, ori_corner_poses):
                if ori_corner_pos[0] < MIDDLE_PIXEL:
                    if diff[0] < 0:
                        dispersion_count += 1
                    else:
                        convergence_count += 1
                else:
                    if diff[0] < 0:
                        convergence_count += 1
                    else:
                        dispersion_count += 1

            if convergence_count > dispersion_count and convergence_count > len(good_diffs) // 2:
                useful_sector = 'backward'
            if dispersion_count > convergence_count and dispersion_count > len(good_diffs) // 2:
                useful_sector = 'forward'
            
        # =================================
        # show sector via opencv
        if show:
            sector_array = np.zeros((400, 400, 3), dtype=np.uint8)
            for key in sector:
                sector_array[400-sector[key]:400, 10+key*90:10+(1+key)*90] = 255
                cv2.putText(
                    sector_array, 
                    str(sector[key]), 
                    (40+key*90, 400-sector[key]-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1,
                    (255, 255, 255),
                    3)
            cv2.imshow('sector_val', sector_array)

        return useful_sector

    def handle_direction_plane_detect(self, good_diffs, ori_corner_poses, show=False):
        DIR_DICT = {
            0: 'left',
            1: 'up',
            2: 'right',
            3: 'down'
        }
        MIDDLE_PIXEL = 320
        SECTOR_FORWARD_BACKWARD_THRESHOLD = 2
        SECTOR_FORWARD_BACKWARD_THRESHOLD2 = 10
        N_GOOD_DIFF_THRESHOLD = 10

        # need enough good_diffs to handle
        if len(good_diffs) < N_GOOD_DIFF_THRESHOLD:
            return None

        # ==============================================
        # get diff section 
        sector = {idx: 0 for idx in range(4)}
        for diff in good_diffs:
            angle = np.arctan2(diff[1], diff[0])
            if angle < 0:
                angle += 2 * np.pi
            
            if angle < np.pi / 8 or angle > 15 * np.pi / 8:
                sector[0] += 1
            elif angle > np.pi / 8 and angle < 7 * np.pi / 8:
                sector[1] += 1
            elif angle > 7 * np.pi / 8 and angle < 9 * np.pi / 8:
                sector[2] += 1
            elif angle > 9 * np.pi / 8 and angle < 15 * np.pi / 8:
                sector[3] += 1
    
        # ==============================================
        # handle four diection (left right up down)
        n_good_diff = len(good_diffs)
        useful_sector = None
        for key in sector:
            if sector[key] > n_good_diff // 2:
                useful_sector = DIR_DICT[key]

        # ==============================================
        # handle forward backward 
        #   (it will override the four direction)
        sector_count = 0
        for key in sector:
            if sector[key] > SECTOR_FORWARD_BACKWARD_THRESHOLD:
                sector_count += 1
        
        if sector[0] > SECTOR_FORWARD_BACKWARD_THRESHOLD2 and \
           sector[2] > SECTOR_FORWARD_BACKWARD_THRESHOLD2 and \
           sector_count > 2:
            
            # get diff convergence & dispersion
            convergence_count = 0
            dispersion_count = 0
            for diff, ori_corner_pos in zip(good_diffs, ori_corner_poses):
                if ori_corner_pos[0] < MIDDLE_PIXEL:
                    if diff[0] < 0:
                        dispersion_count += 1
                    else:
                        convergence_count += 1
                else:
                    if diff[0] < 0:
                        convergence_count += 1
                    else:
                        dispersion_count += 1

            if convergence_count > dispersion_count and convergence_count > len(good_diffs) // 2:
                useful_sector = 'backward'
            if dispersion_count > convergence_count and dispersion_count > len(good_diffs) // 2:
                useful_sector = 'forward'
            
        # =================================
        # show sector via opencv
        if show:
            sector_array = np.zeros((400, 400, 3), dtype=np.uint8)
            for key in sector:
                sector_array[400-sector[key]:400, 10+key*90:10+(1+key)*90] = 255
                cv2.putText(
                    sector_array, 
                    str(sector[key]), 
                    (40+key*90, 400-sector[key]-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1,
                    (255, 255, 255),
                    3)
            cv2.imshow('sector_val', sector_array)

        return useful_sector

    def direction_filter(self, direction, agent_action):
        if isinstance(agent_action['forward'], np.ndarray):
            forward_action = int(agent_action['forward'][0])
        else:
            forward_action = agent_action['forward']
        if isinstance(agent_action['jump'], np.ndarray):
            jump_action = int(agent_action['jump'][0])
        else:
            jump_action = agent_action['jump']

        # prehandle direction via agent action
        if forward_action == 0 and jump_action == 0:
            direction = None

        return direction