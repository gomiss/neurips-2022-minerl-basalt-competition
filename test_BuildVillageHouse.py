from run_agent import main as run_agent_main
from buildVillageHouse.run_agent import Noinv
from config import EVAL_EPISODES, EVAL_MAX_STEPS
import os

def unzipmodels():
    base_dir = os.path.dirname(__file__)
    hub_dir = os.path.join(base_dir, "hub")
    ckpt_dir = os.path.join(hub_dir, "checkpoints")
    model_dir = os.path.join(base_dir, "buildVillageHouse", "models")
    # print(base_dir,hub_dir,ckpt_dir,model_dir)
    if not os.path.exists(os.path.join(model_dir, 'flattenareaV4resnet50.pt')) or not os.path.exists(os.path.join(model_dir, 'MineRLBasaltMovingNoinv.weights')):
        # print(base_dir)

        # print(f"hub dir: {hub_dir}, ckpt dir: {ckpt_dir}, model dir: {model_dir}")
        os.system(f"""cd {hub_dir} && 7z x myfiles.zip.001 -y && mv {os.path.join(ckpt_dir,"flattenareaV4resnet50.pt")} {model_dir}  && mv {os.path.join(ckpt_dir,"MineRLBasaltMovingNoinv.weights")} {model_dir}""")


def main():

    # unzipmodels()
    base_dir = os.path.dirname(__file__)
    model_dir = os.path.join(base_dir, "buildVillageHouse", "models")
    Noinv(
        model = os.path.join(model_dir,"foundation-model-1x.model"),
        weights=os.path.join(base_dir, "train", "MineRLBasaltMovingNoinv.weights"),
        env="MineRLBasaltBuildVillageHouse-v0",   
        n_episodes=EVAL_EPISODES,
        max_steps=EVAL_MAX_STEPS
    )


if __name__ == "__main__":
    main()