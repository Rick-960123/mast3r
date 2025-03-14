import os
import torch
import tempfile

from mast3r.demo import get_args_parser, main_demo, get_reconstructed_scene
from mast3r.utils.misc import hash_md5
from mast3r.model import AsymmetricMASt3R
from dust3r.demo import set_print_with_timestamp
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as pl
pl.ion()
torch.backends.cuda.matmul.allow_tf32 = True 
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    set_print_with_timestamp()

    if args.tmp_dir is None:
        args.tmp_dir = "/root/mast3r/mast3r_gradio_demo"
        chkpt_tag = hash_md5(args.weights)
        os.makedirs(args.tmp_dir, exist_ok=True)
        cache_path = os.path.join(args.tmp_dir, chkpt_tag)
        os.makedirs(cache_path, exist_ok=True)
        tempfile.tempdir = cache_path

    model = AsymmetricMASt3R.from_pretrained(args.weights).to(args.device)
    

    filelist = []
    img_names = []
    for dirpath, dirnames, filenames in os.walk("/root/Datasets/slam2000-雪乡情-正走/colmap/images/dust3r"):
        for filename in filenames:
            if filename.endswith(".png"):
                filelist.append(os.path.join(dirpath, filename))
                img_names.append(filename)
    poses_dict = {}
    with open('/root/Datasets/slam2000-雪乡情-正走/colmap/sparse/0/images.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            if line.strip():
                img_name = line.strip().split(' ')[-1]
                pose_list = [float(x) for x in line.strip().split(' ')[1:8]]
                pose = np.eye(4)
                pose[:3, :3] = Rotation.from_quat(np.array([pose_list[1], pose_list[2], pose_list[3], pose_list[0]])).as_matrix()
                pose[:3, 3] = np.array(pose_list[4:7])
                pose = np.linalg.inv(pose)
                poses_dict[img_name] = pose

    poses = {}
    for i, img_name in enumerate(img_names):
        poses[filelist[i]] = {}
        poses[filelist[i]]['cam2w'] = torch.from_numpy(poses_dict[img_name]).to(args.device).to(torch.float32)

    get_reconstructed_scene(args.tmp_dir, False, model, args.device, args.silent, args.image_size, None, filelist, "refine+depth", 0.07, 500, 0.014, 200, 1.5, 5.0, True, False, True, False, 0.05, "swin", 5, False, 0, 0,True, init=poses)
