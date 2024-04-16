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

class Text2MotionAlignmentDataset(data.Dataset):
    def __init__(self, opt, t2m_split_file, meta_dir=None):
        super(Text2MotionAlignmentDataset, self).__init__()
        self.opt = opt
        self.max_seq_length = 200
        amass_id_list = []
        with cs.open(t2m_split_file, "r") as f:
            for line in f.readlines():
                amass_id_list.append(line.strip())
        
        self.data_dict = {}
        self.name_list = {}
        self.data_dict["t2m"], self.name_list["t2m"] = self.read_all_t2m_data(amass_id_list)
    
    def __len__(self):
        return len(self.data_dict["t2m"])

    def __getitem__(self, item):
        
        batch = self.read_one_t2m_data("t2m", item)
        return batch
    
    def read_all_t2m_data(self, id_list):
        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list, desc="Reading text-to-motion dataset"):
            try:
                body = np.load(pjoin(self.opt["t2m_motion_dir"], name+".npy"))
                if body.shape[0] < 40:
                    continue
                with cs.open(pjoin(self.opt["text_dir"], name+".txt")) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()
                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag
                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)
                                data_dict[new_name] = {'text':[text_dict], "body": body}
                                new_name_list.append(new_name)
                        except:
                            pass
                
                if flag:
                    data_dict[name] = {'text': text_data, 'body': body}
                    new_name_list.append(name)
            
            except:
                pass
        
        return data_dict, new_name_list
    
    def read_one_t2m_data(self, key, item):
        if item < len(self.name_list[key]):
            index = item
        else:
            index = item % len(self.name_list[key])
        name = self.name_list[key][index]
        body = self.data_dict[key][name]["body"]
        text_list = self.data_dict[key][name]["text"]
        text_data = random.choice(text_list)    # Random sample one text
        caption = text_data["caption"]
        t_tokens = text_data["tokens"]
        mot_len = body.shape[0]
        if mot_len > self.opt["window_size"][key]:
            # If motion sequence is longer than target lenght, we randomly select N frames from it. 
            # This approach partially acts as data augmentation during training.
            # indices = np.random.choice(mot_len, (self.opt["window_size"][key], ), replace=False)
            i = np.random.randint(0, mot_len - self.opt["window_size"][key])
            j = i + self.opt["window_size"][key]
            # indices = np.sort(indices)
            body = body[i:j]
            mot_len = body.shape[0]
        elif mot_len < self.opt["window_size"][key]:
            # If the motion sequence is shorter than target length, we pad zero poses.
            pad_len = self.opt["window_size"][key] - mot_len
            pad_pose = np.zeros((pad_len, body.shape[1]))
            body = np.concatenate([body, pad_pose], axis=0)

        return body, caption, mot_len
    
    
DATASET_MAP = {
    "Text2MotionAlignmentDataset": Text2MotionAlignmentDataset, 
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