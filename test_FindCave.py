from findcave.run_agent import main as run_agent_main
from config import EVAL_EPISODES, EVAL_MAX_STEPS

def main():
    run_agent_main(
        vpt_model=r"train/foundation-model-1x.model",
        weights="train/MineRLBasaltFindCave.weights",
        yolo_weights='train/findcave_detector.pt',
        env="MineRLBasaltFindCave-v0",
        n_episodes=EVAL_EPISODES,
        max_steps=EVAL_MAX_STEPS
    )

def main4retrain_phase():
    run_agent_main(
        vpt_model=r"data/VPT-models/foundation-model-1x.model",
        weights="train/MineRLBasaltFindCave.weights",
        yolo_weights='train/findcave_detector.pt',
        env="MineRLBasaltFindCave-v0",
        n_episodes=EVAL_EPISODES,
        max_steps=EVAL_MAX_STEPS
    )

if __name__ == "__main__":
    main()