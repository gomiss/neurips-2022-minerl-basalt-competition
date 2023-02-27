from findcave.run_agent import main as run_agent_main
from config import EVAL_EPISODES, EVAL_MAX_STEPS
import sys
import os
savedStdout = sys.stdout  #保存标准输出流

def main():
    for i in range(1):
        if not os.path.exists('debug_sequence'+str(i)):
            os.mkdir('debug_sequence'+str(i))
        with open('debug_sequence'+str(i) + '/out.txt', 'w+') as file:
            sys.stdout = file  # 标准输出重定向至文件
            run_agent_main(
                vpt_model=r"findcave/findcave_models/foundation-model-1x.model",
                weights="findcave/findcave_models/MineRLBasaltFindCave.weights",
                yolo_weights='findcave/findcave_models/best.pt',
                env="MineRLBasaltFindCave-v0",
                n_episodes=1,
                max_steps=EVAL_MAX_STEPS,
                now_i=i
            )

if __name__ == "__main__":
    main()
    sys.stdout = savedStdout  # 恢复标准输出流
    print('This message is for screen!')