from animalpen.run_agent_for_train import main as run_agent_main
from config import EVAL_EPISODES, EVAL_MAX_STEPS

def main():
    run_agent_main(
        model="data/VPT-models/2x.model",
        weights="data/VPT-models/foundation-model-2x.weights",
        env="MineRLBasaltCreateVillageAnimalPen-v0",
        n_episodes=EVAL_EPISODES,
        max_steps=EVAL_MAX_STEPS,
        show=True
    )

if __name__ == '__main__':
    main()