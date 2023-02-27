# Train one model for each task
import os,sys
sys.path.append(os.path.dirname(__file__))

from buildvillagehouse_behavioural_cloning import behavioural_cloning_train
import os,sys
sys.path.append(os.path.dirname(__file__))

def train_bc():
    village_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'MineRLBasaltBuildVillageHouse-v0')
    animalpen_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'MineRLBasaltCreateVillageAnimalPen-v0')
    cave_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'MineRLBasaltFindCave-v0')

    in_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'VPT-models', 'foundation-model-1x.model')

    in_weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'VPT-models', 'foundation-model-1x.weights')

    out_weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'train', 'MineRLBasaltMovingNoinv.weights')


    print("===Training Moving model===")
    behavioural_cloning_train(
        data_dir=[
                 animalpen_dir,
                 cave_dir,
                 village_data_dir
                 ],
        in_model=in_model_dir,
        in_weights=in_weights_dir,
        out_weights=out_weights_dir
    )

if __name__ == "__main__":
    train_bc()
