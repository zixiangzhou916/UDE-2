import os
# import sys
# sys.path.append(os.getcwd())
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import numpy as np
import random
from os.path import join as pjoin
import random
import json
import copy
import codecs as cs
from tqdm import tqdm
from scipy import ndimage
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class MotionAutoEncoderDataset(data.Dataset):
    def __init__(self, opt, t2m_split_file, a2m_split_file, s2m_split_file, meta_dir=None):
        super(MotionAutoEncoderDataset, self).__init__()
        self.opt = opt
        amass_id_list = []
        aist_id_list = []
        beat_id_list = []
        with cs.open(t2m_split_file, "r") as f:
            for line in f.readlines():
                amass_id_list.append(line.strip())
        
        with cs.open(a2m_split_file, "r") as f:
            for line in f.readlines():
                aist_id_list.append(line.strip())

        with cs.open(s2m_split_file, "r") as f:
            for line in f.readlines():
                beat_id_list.append(line.strip()) 

        self.data = []
        self.lengths = []
        for name in tqdm(amass_id_list):
            try:
                motion = np.load(pjoin(opt["amass_motion_dir"], name + '.npy'))
                if motion.shape[0] < 40:
                    continue
                self.data.append(motion)
            except:
                pass

        for name in tqdm(aist_id_list):
            try:
                data = np.load(pjoin(opt["aist_motion_dir"], name + '.npy'), allow_pickle=True).item()
                motion = data["motion_smpl"]

                # downsample, the target fps is 20, and the original fps is 60
                motion = motion[::2]
                self.data.append(motion)

            except:
                pass
                
        for name in tqdm(beat_id_list):
            try:
                data = np.load(pjoin(opt["beat_motion_dir"], name + '.npy'), allow_pickle=True).item()
                motion = data["body"]

                # downsample, the target fps is 20, and the original fps is 60
                motion = motion[::4]
                self.data.append(motion)

            except:
                pass
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        
        motion = self.data[item]
        seq_len = motion.shape[0]
        if seq_len < self.opt["window_size"]:
            pad_motion = np.zeros((self.opt["window_size"]-seq_len, motion.shape[1]))
            motion = np.concatenate([motion, pad_motion], axis=0)
        elif seq_len > self.opt["window_size"]:
            idx = np.random.randint(0, seq_len-self.opt["window_size"])
            motion = motion[idx:idx+self.opt["window_size"]]
            seq_len = self.opt["window_size"]
        return motion, seq_len

    @staticmethod
    def resize_motion(input_motion, target_length):
        nframes = input_motion.shape[0]
        x = np.arange(0, target_length)
        x = np.random.choice(target_length, size=(nframes,), replace=False)
        x = np.sort(x)
        if x[0] != 0: x[0] = 0
        if x[-1] != target_length - 1: x[-1] = target_length - 1
        new_x = np.arange(0, target_length)
        
        # Interpolate translation
        transl = input_motion[..., :3]
        new_transl = []
        for i in range(transl.shape[-1]):
            f = interpolate.interp1d(x, transl[:, i])
            new_transl_ = f(new_x)
            new_transl.append(new_transl_)
        new_transl = np.stack(new_transl, axis=-1)
        
        # Interpolate rotvec
        rotvecs = input_motion[..., 3:]
        rotvecs = np.reshape(rotvecs, newshape=(nframes, -1, 3))    # [T, J, 3]
        new_rotvecs = []
        for j in range(rotvecs.shape[1]):
            y = rotvecs[:, j]   # [T, 3]
            slerp = Slerp(x, R.from_rotvec(y))
            out = slerp(new_x)
            new_y = out.as_rotvec() # [T, 3]
            new_rotvecs.append(new_y)
        new_rotvecs = np.stack(new_rotvecs, axis=1) # [T, J, 3]
        new_rotvecs = np.reshape(new_rotvecs, newshape=(target_length, -1)) # [T, 3*J]
        return np.concatenate([new_transl, new_rotvecs], axis=-1)
    
DATASET_MAP = {
    "MotionAutoEncoderDataset": MotionAutoEncoderDataset, 
}

def get_dataloader(data_conf, loader_conf, meta_dir=None):
    split_files = {}
    for key, item in loader_conf["split_path"].items():
        if isinstance(loader_conf["split"][key], str):
            split_files["{:s}_split_file".format(key)] = pjoin(item, loader_conf["split"][key]+".txt")
        elif isinstance(loader_conf["split"][key], list):
            split_files["{:s}_split_file".format(key)] = [pjoin(item, split+".txt") for split in loader_conf["split"][key]]
    dataset = DATASET_MAP[loader_conf["dataset"]](data_conf, **split_files, meta_dir=meta_dir)
    loader = DataLoader(dataset, 
                        batch_size=loader_conf["batch_size"], 
                        drop_last=True, 
                        num_workers=loader_conf["workers"], 
                        shuffle=loader_conf["shuffle"], 
                        pin_memory=True)
    return loader, dataset    

if __name__ == "__main__":
    import yaml
    with open("configs/perception/config_vae_exp1.yaml", "r") as f:
        conf = yaml.safe_load(f)
    
    loader, _ = get_dataloader(conf["data"]["dataset"], conf["data"]["loader"]["vald"])
    for _, batch in enumerate(loader):
        print(batch[1])

