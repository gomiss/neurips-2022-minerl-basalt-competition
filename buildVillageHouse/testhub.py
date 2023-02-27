import os

def unzipflattenmodel(base_dir):
    hub_dir = os.path.join(base_dir, "hub")
    ckpt_dir = os.path.join(hub_dir, "checkpoints")
    model_dir = os.path.join(base_dir, "models")
    # print(f"hub dir: {hub_dir}, ckpt dir: {ckpt_dir}, model dir: {model_dir}")
    os.system(f"""cd {hub_dir} && 7z x myfiles.zip.001 -y && mv {os.path.join(ckpt_dir,"flattenareaV4resnet50.pt")} {model_dir}  && mv {os.path.join(ckpt_dir,"MineRLBasaltMovingNoinv.pt")} {model_dir}""")

# def unzipVPTmodel(base_dir):
#     hub_dir = os.path.join(base_dir, "hub")
#     ckpt_dir = os.path.join(hub_dir, "checkpoints")
#     model_dir = os.path.join(base_dir, "models")
#     # print(f"hub dir: {hub_dir}, ckpt dir: {ckpt_dir}, model dir: {model_dir}")
#     os.system(f"""cd {hub_dir} && 7z x MineRLBasaltMovingNoinv.zip.001 -y && mv {os.path.join(ckpt_dir,"MineRLBasaltMovingNoinv.weights")} {model_dir}""")

def unzipmodels():
    base_dir = os.path.dirname(__file__)
    # print(base_dir)
    if not os.path.exists(os.path.join(base_dir, 'models', 'flattenareaV4resnet50.pt')):
        # print(base_dir)
        unzipflattenmodel(base_dir)

    if not os.path.exists(os.path.join(base_dir, 'models', 'MineRLBasaltMovingNoinv.weights')):
        # print(base_dir)
        unzipflattenmodel(base_dir) 

if __name__ == '__main__':
    unzipmodels()