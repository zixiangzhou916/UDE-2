import os, sys
sys.path.append(os.getcwd())
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import numpy as np
import os
import random
from os.path import join as pjoin
import random
import json
import codecs as cs
from tqdm import tqdm
from scipy import ndimage
import importlib

from torch.utils.data._utils.collate import default_collate
# from dataloader.ude.word_vectorizer import WordVectorizerV2

SPEAKER_ID = {
    "wayne": 0, "kieks": 1, "nidal": 2, "zhao": 3, "lu": 4,
    "zhang": 5, "carlos": 6, "jorge": 7, "itoi": 8, "daiki": 9,
    "jaime": 10, "scott": 11, "li": 12, "ayana": 13, "luqi": 14,
    "hailing": 15, "kexin": 16, "goto": 17, "reamey": 18, "yingqing": 19,
    "tiffnay": 20, "hanieh": 21, "solomon": 22, "katya": 23, "lawrence": 24,
    "stewart": 25, "carla": 26, "sophie": 27, "catherine": 28, "miranda": 29, 
    "ChenShuiRuo": 0
}

SEMANTIC_ID = {
    "deictic_l": 0, "metaphoric_m": 1, "iconic_h": 2, "metaphoric_l": 3,
    "beat_align": 4, "metaphoric_h": 5, "deictic_h": 6, "iconic_m": 7,
    "nogesture": 8, "deictic_m": 9, "need_cut": 10, "iconic_l": 11, "habit": 12 
}

EMOTION_ID = [0, 1, 2, 3, 4, 5, 6, 7]

RECORDING_TYPE = {
    0: "English Speech", 
    1: "English Conversation", 
    2: "Chinese Speech", 
    3: "Chinese Conversation", 
    4: "Spanish Speech", 
    5: "Spanish Conversation", 
    6: "Japanese Speech", 
    7: "Japanese Conversation"
}

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def normalize_trans(motion):
    """
    :param motion: [num_frames, num_dims]
    """
    glob_trans = motion[:, :3]  # [num_frames, 3]
    avg_trans = np.mean(glob_trans, axis=0, keepdims=True)  # [1, 3]
    motion[:, :3] -= avg_trans
    return motion

""" VQ-VAE dataloader (HumanML3D) """
class HumanML3DTokenizerDataset(data.Dataset):
    def __init__(self, opt, t2m_split_file, meta_dir=None,**kwargs):
        super(HumanML3DTokenizerDataset, self).__init__()
        self.opt = opt
        id_list = []
        with cs.open(t2m_split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
                
        # Read all data
        self.data = []
        self.lengths = []
        for name in tqdm(id_list, desc="Loading all data"):
            try:
                # print(pjoin(self.opt["motion_dir"], name+".npy"))
                motion = np.load(pjoin(self.opt["motion_dir"], name+".npy"))    # fps = 20
                if motion.shape[0] < opt["window_size"]: continue
                self.lengths.append(motion.shape[0] - opt["window_size"])
                self.data.append(motion)
            except:
                pass
        
        self.cumsum = np.cumsum([0] + self.lengths)

    def __len__(self):
        return self.cumsum[-1]
    
    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0

        motion = self.data[motion_id][idx:idx+self.opt["window_size"]]
        return {"body": motion}
    
class AISTPPTokenizerDataset(data.Dataset):
    def __init__(self, opt, a2m_split_file, meta_dir=None,**kwargs):
        super(AISTPPTokenizerDataset, self).__init__()
        self.opt = opt
        id_list = []
        with cs.open(a2m_split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        
        # Read all data
        self.data = []
        self.lengths = []
        for name in tqdm(id_list, desc="Loading all data"):
            try:
                # print(pjoin(self.opt["motion_dir"], name+".npy"))
                data = np.load(pjoin(self.opt["motion_dir"], name+".npy"), allow_pickle=True).item()    # fps = 60
                motion = data["motion_smpl"][::2]   # downsample to fps = 30
                if motion.shape[0] < opt["window_size"]: continue
                self.lengths.append(motion.shape[0] - opt["window_size"])
                self.data.append(motion)
            except:
                pass
        
        self.cumsum = np.cumsum([0] + self.lengths)
        
    def __len__(self):
        return self.cumsum[-1]
    
    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0

        motion = self.data[motion_id][idx:idx+self.opt["window_size"]]
        return {"body": motion}
    
class BEATTokenizerDataset(data.Dataset):
    def __init__(self, opt, s2m_split_file, meta_dir):
        super(BEATTokenizerDataset, self).__init__()
        self.opt = opt
        
        if "beat_id_list" in opt.keys():
            target_id_list = opt["beat_id_list"]        # ID list to train
        else:
            target_id_list = [i for i in range(30)]     # ID list to train
        beat_id_list = []
        with cs.open(s2m_split_file, "r") as f:
            for line in f.readlines():
                name = line.strip()
                id = int(name.split("_")[0])
                if id in target_id_list:
                    beat_id_list.append(line.strip())
        
        self.data = []
        self.lengths = []
        for name in tqdm(beat_id_list, desc="Loading all data"):
            try:
                # print(pjoin(opt["motion_dir"], name+".npy"))
                data = np.load(pjoin(opt["motion_dir"], name+".npy"), allow_pickle=True).item()
                
                body = data["body"]
                left = data["left"]
                right = data["right"]
                expr = data["expression"]
                body = body[::4]    # Downsample to 30fps
                left = left[::4]    # Downsample to 30fps
                right = right[::4]    # Downsample to 30fps
                expr = expr[::2]

                if body.shape[0] < opt["window_size"]:
                    continue
                
                length = min(body.shape[0], expr.shape[0])
                self.lengths.append(length - opt["window_size"])
                self.data.append(
                    {
                        "body": body, "left": left, "right": right, "expr": expr
                    }
                )
            
            except:
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

    def __len__(self):
        return self.cumsum[-1]
    
    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        
        data = self.data[motion_id]
        body = data["body"][idx:idx+self.opt["window_size"]]
        left = data["left"][idx:idx+self.opt["window_size"]]
        right = data["right"][idx:idx+self.opt["window_size"]]
        expr = data["expr"][idx:idx+self.opt["window_size"]]
        
        batch = {
            "body": body, "left": left, "right": right, "expr": expr
        }
        
        return batch
    
""" UDE dataloader """
class UDEDataset(data.Dataset):
    def __init__(
        self, opt, 
        t2m_split_file=None, 
        a2m_split_file=None, 
        s2m_split_file=None, 
        flame_split_file=None, 
        meta_dir=None, w_vectorizer=None
    ):
        super(UDEDataset, self).__init__()
        self.w_vectorizer = w_vectorizer
        self.opt = opt
        self.times = opt["times"]
        id_lists = {}
        # Text-to-Motion
        if "t2m" in self.opt["modality"]:
            id_lists["t2m"] = []
            with cs.open(t2m_split_file, "r") as f:
                for line in f.readlines():
                    id_lists["t2m"].append(line.strip())
        # Audio-to-Motion
        if "a2m" in self.opt["modality"]:
            id_lists["a2m"] = []
            with cs.open(a2m_split_file, "r") as f:
                for line in f.readlines():
                    id_lists["a2m"].append(line.strip())
        # Speech-to-Motion
        if "s2m" in self.opt["modality"]:
            id_lists["s2m"] = []
            if "beat_id_list" in opt.keys():
                target_id_list = opt["beat_id_list"]        # ID list to train
            else:
                target_id_list = [i for i in range(30)]     # ID list to train
            with cs.open(s2m_split_file, "r") as f:
                for line in f.readlines():
                    name = line.strip()
                    id = int(name.split("_")[0])
                    if id in target_id_list:
                        id_lists["s2m"].append(name)
            
            with open(self.opt["beat_vocab_dir"], "r") as f:
                self.lang_model = json.load(f)
        
        # Load all dataset
        self.data_dict = {}
        self.name_list = {}
        for key, id_list in id_lists.items():
            if "t2m" == key:
                self.data_dict["t2m"], self.name_list["t2m"] = self.read_all_t2m_data(id_list)
            elif "a2m" == key:
                self.data_dict["a2m"], self.name_list["a2m"] = self.read_all_a2m_data(id_list)
            elif "s2m" == key:
                self.data_dict["s2m"], self.name_list["s2m"] = self.read_all_s2m_data(id_list)
        
        # Make sure they have the same lenght, if not, we duplicate the shorter ones
        max_length = np.max([len(item) for _, item in self.data_dict.items()])    
        self.num_data = max_length
        
    def __len__(self):
        return self.num_data * self.times
    
    def __getitem__(self, item):
        
        batch = {}
        for key in self.data_dict.keys():
            if "t2m" in key:
                batch[key] = self.read_one_t2m_data(key, item)
            elif "a2m" in key:
                batch[key] = self.read_one_a2m_data(key, item)
            elif "s2m" in key:
                batch[key] = self.read_one_s2m_data(key, item)
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
    
    def read_all_a2m_data(self, id_list):
        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list, desc="Reading audio-to-motion dataset"):
            try:
                data = np.load(pjoin(self.opt["a2m_motion_dir"], name+".npy"), allow_pickle=True).item()
                body = data["motion_smpl"][::2] # downsample to fps=30
                audio = data["audio_sequence"]
                duration = data["duration"]
                data_dict[name] = {"audio": audio, "body": body, "name": name, "duration": duration}
                new_name_list.append(name)
            except:
                pass
        return data_dict, new_name_list
        
    def read_all_s2m_data(self, id_list):
        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list, desc="Reading speech-to-motion dataset"):
            try:
                data = np.load(pjoin(self.opt["s2m_motion_dir"], name+".npy"), allow_pickle=True).item()
                body = data["body"][::4]        # Downsample to fps=30
                left = data["left"][::4]        # Downsample to fps=30
                right = data["right"][::4]      # Downsample to fps=30
                expr = data["expression"][::2]  # Downsample to fps=30
                audio = data["audio"]           # fps=30
                duration = audio.shape[0] / 16000
                
                text = data["text"]
                word = data["word"]             # fps=30
                semantic = self.parse_semantic_label(data["semantic"])     # fps=30
                emotion = self.parse_emotion_label(data["emotion"])       # fps=30
                speaker_id = SPEAKER_ID[name.split("_")[1]]
                
                length = np.min([body.shape[0], expr.shape[0], int(duration * 30), len(word), semantic.shape[0], emotion.shape[0]])
                audio_length = int(length / 30 * 16000) + 1
                new_name_list.append(name)
        
                data_dict[name] = {
                    "body": body[:length], "left": left[:length], "right": right[:length], "expr": expr[:length], 
                    "word": word[:length], "audio": audio[:audio_length], "semantic": semantic[:length], 
                    "emotion": emotion[:length], "speaker_id": speaker_id, "name": name
                }
            except:
                pass
        return data_dict, new_name_list
    
    @staticmethod
    def parse_semantic_label(semantic):
        semantic_label = []
        for sem in semantic:
            semantic_label.append(SEMANTIC_ID[sem[0]])
        return np.asarray(semantic_label, dtype=int)
    
    @staticmethod
    def parse_emotion_label(emotion):
        return np.asarray(emotion, dtype=int)
    
    def parse_word_tokens(self, words):
        tokens = []
        for i, w in enumerate(words):
            w = w.replace(",", "").replace(".", "").replace("?", "").replace("!", "")
            if w == "":
                tokens.append(self.lang_model["PAD"])
                # tokens.append(self.lang_model.PAD_token)
            elif w in self.lang_model.keys():
                tokens.append(self.lang_model[w])
                # tokens.append(self.lang_model.get_word_index(w))
            else:
                tokens.append(self.lang_model["UNK"])
        tokens = np.asarray(tokens, dtype=int)
        
        # Get the caption
        unique_tokens, unique_index = np.unique(tokens, return_index=True)
        unique_index = np.sort(unique_index)
        caption = " ".join([words[i] for i in unique_index])
        return caption, tokens
    
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
        
        # Get word tokens from Vocab
        if len(t_tokens) < self.opt["t2m_max_text_len"]:
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
            t_tokens += ['unk/OTHER'] * (self.opt["t2m_max_text_len"] + 2 - sent_len)
        else:
            t_tokens = t_tokens[:self.opt["t2m_max_text_len"]]
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
        # word_tokens = []
        # for i, t_tok in enumerate(t_tokens):
        #     word_emb, _, tok_id = self.w_vectorizer[t_tok]
        #     if i >= sent_len:
        #         word_tokens.append(self.opt["t2m_txt_pad_idx"])
        #     else:
        #         word_tokens.append(tok_id)
        # word_tokens = np.array(word_tokens, dtype=int)
        
        batch = {"body": body, "text": caption, "length": mot_len}
        return batch
    
    def read_one_a2m_data(self, key, item):
        if item < len(self.name_list[key]):
            index = item
        else:
            index = item % len(self.name_list[key])
        
        name = self.name_list[key][index]
        body = self.data_dict[key][name]["body"] # The motion has already been downsampled to fps=30
        audio = self.data_dict[key][name]["audio"] # The fps of audio is 60 by default, and we don't modify it.
        duration = self.data_dict[key][name]["duration"]
        mot_len = body.shape[0]
        if mot_len > self.opt["window_size"][key]:
            m_start_idx = np.random.randint(0, mot_len - self.opt["window_size"][key])
            m_end_idx = m_start_idx + self.opt["window_size"][key]
            a_start_idx = int(m_start_idx / 30 * 16000)
            a_end_idx = a_start_idx + int(self.opt["window_size"][key] / 30 * 16000)
            body = body[m_start_idx:m_end_idx]
            audio = audio[a_start_idx:a_end_idx]
        else:
            raise ValueError("Sequence length is not long enough!")
        
        mot_len = body.shape[0]
        batch = {
            "body": body, "audio": audio, "length": mot_len, 
            "name": "{:s}_{:d}_{:d}".format(name, a_start_idx, a_end_idx)
        }
        return batch
    
    def read_one_s2m_data(self, key, item):

        if item < len(self.name_list[key]):
            index = item
        else:
            index = item % len(self.name_list[key])
        name = self.name_list[key][index]
        data = self.data_dict[key][name]
        mot_len = data["body"].shape[0]
        m_start_idx = np.random.randint(0, mot_len - self.opt["window_size"][key])
        m_end_idx = m_start_idx + self.opt["window_size"][key]
        a_start_idx = int(m_start_idx / 30 * 16000)
        a_end_idx = a_start_idx + int(self.opt["window_size"][key] / 30 * 16000)

        body = data["body"][m_start_idx:m_end_idx]
        left = data["left"][m_start_idx:m_end_idx]
        right = data["right"][m_start_idx:m_end_idx]
        audio = data["audio"][a_start_idx:a_end_idx]
        caption, words = self.parse_word_tokens(data["word"][m_start_idx:m_end_idx])
        semantic = data["semantic"][m_start_idx:m_end_idx]
        emotion = self.parse_emotion_label(data["emotion"][m_start_idx:m_end_idx])
        speaker_id = np.array(data["speaker_id"], dtype=int)
        # name = data["name"]
        
        mot_len = body.shape[0]
        batch = {
            "body": body, "left": left, "right": right, 
            "audio": audio, "word": words, "caption": caption, 
            "semantic": semantic, "emotion": emotion, 
            "speaker_id": speaker_id, "name": "{:s}_{:d}_{:d}".format(name, m_start_idx, m_end_idx), 
            "length": mot_len
        }
        return batch

""" UDE dataloader for evaluation """
class UDEDatasetEval(data.Dataset):
    def __init__(
        self, opt, 
        t2m_split_file=None, 
        a2m_split_file=None, 
        s2m_split_file=None, 
        flame_split_file=None, 
        meta_dir=None, w_vectorizer=None
    ):
        super(UDEDatasetEval, self).__init__()
        self.w_vectorizer = w_vectorizer
        self.opt = opt
        self.times = opt["times"]
        id_lists = {}
        # Text-to-Motion
        if "t2m" in self.opt["modality"]:
            id_lists["t2m"] = []
            with cs.open(t2m_split_file, "r") as f:
                for line in f.readlines():
                    id_lists["t2m"].append(line.strip())
        # Audio-to-Motion
        if "a2m" in self.opt["modality"]:
            id_lists["a2m"] = []
            with cs.open(a2m_split_file, "r") as f:
                for line in f.readlines():
                    id_lists["a2m"].append(line.strip())
        # Speech-to-Motion
        if "s2m" in self.opt["modality"]:
            id_lists["s2m"] = []
            if "beat_id_list" in opt.keys():
                target_id_list = opt["beat_id_list"]        # ID list to train
            else:
                target_id_list = [i for i in range(30)]     # ID list to train
            with cs.open(s2m_split_file, "r") as f:
                for line in f.readlines():
                    name = line.strip()
                    id = int(name.split("_")[0])
                    if id in target_id_list:
                        id_lists["s2m"].append(name)
            
            with open(self.opt["beat_vocab_dir"], "r") as f:
                self.lang_model = json.load(f)
        
        # Load all dataset
        self.data_dict = []
        for key, id_list in id_lists.items():
            if "t2m" == key:
                self.data_dict += self.read_all_t2m_data(id_list)
            elif "a2m" == key:
                self.data_dict += self.read_all_a2m_data(id_list)
            elif "s2m" == key:
                self.data_dict += self.read_all_s2m_data(id_list)
                
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, item):
        data = self.data_dict[item]
        task = data["task"]
        if task == "t2m":
            batch = self.read_one_t2m_data(data=data)
        elif task == "a2m":
            batch = self.read_one_a2m_data(data=data)
        elif task == "s2m":
            batch = self.read_one_s2m_data(data=data)
        return batch
    
    def read_all_t2m_data(self, id_list):
        new_name_list = []
        data_dict = []
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
                                data_dict.append({'task': 't2m', 'text':[text_dict], "body": body})
                                new_name_list.append(new_name)
                        except:
                            pass
                
                if flag:
                    data_dict.append({'task': 't2m', 'text': text_data, 'body': body})
                    new_name_list.append(name)
            
            except:
                pass
        
        return data_dict
    
    def read_all_a2m_data(self, id_list):
        new_name_list = []
        data_dict = []
        for name in tqdm(id_list, desc="Reading audio-to-motion dataset"):
            try:
                data = np.load(pjoin(self.opt["a2m_motion_dir"], name+".npy"), allow_pickle=True).item()
                body = data["motion_smpl"][::2] # downsample to fps=30
                audio = data["audio_sequence"]
                duration = data["duration"]
                data_dict.append({"audio": audio, "body": body, "name": name, "duration": duration, "task": "a2m"})
                new_name_list.append(name)
            except:
                pass
        return data_dict
    
    def read_all_s2m_data(self, id_list):
        new_name_list = []
        data_dict = []
        for name in tqdm(id_list, desc="Reading speech-to-motion dataset"):
            try:
                data = np.load(pjoin(self.opt["s2m_motion_dir"], name+".npy"), allow_pickle=True).item()
                body = data["body"][::4]        # Downsample to fps=30
                left = data["left"][::4]        # Downsample to fps=30
                right = data["right"][::4]      # Downsample to fps=30
                expr = data["expression"][::2]  # Downsample to fps=30
                audio = data["audio"]           # fps=30
                duration = audio.shape[0] / 16000
                
                text = data["text"]
                word = data["word"]             # fps=30
                semantic = self.parse_semantic_label(data["semantic"])     # fps=30
                emotion = self.parse_emotion_label(data["emotion"])       # fps=30
                speaker_id = SPEAKER_ID[name.split("_")[1]]
                
                length = np.min([body.shape[0], expr.shape[0], int(duration * 30), len(word), semantic.shape[0], emotion.shape[0]])
                audio_length = int(length / 30 * 16000) + 1
                new_name_list.append(name)
        
                data_dict.append(
                    {
                        "body": body[:length], "left": left[:length], "right": right[:length], "expr": expr[:length], 
                        "word": word[:length], "audio": audio[:audio_length], "semantic": semantic[:length], 
                        "emotion": emotion[:length], "speaker_id": speaker_id, "name": name, "task": "s2m"
                    }
                )
            except:
                pass
        return data_dict
    
    @staticmethod
    def parse_semantic_label(semantic):
        semantic_label = []
        for sem in semantic:
            semantic_label.append(SEMANTIC_ID[sem[0]])
        return np.asarray(semantic_label, dtype=int)
    
    @staticmethod
    def parse_emotion_label(emotion):
        return np.asarray(emotion, dtype=int)
    
    def parse_word_tokens(self, words):
        tokens = []
        for i, w in enumerate(words):
            w = w.replace(",", "").replace(".", "").replace("?", "").replace("!", "")
            if w == "":
                tokens.append(self.lang_model["PAD"])
                # tokens.append(self.lang_model.PAD_token)
            elif w in self.lang_model.keys():
                tokens.append(self.lang_model[w])
                # tokens.append(self.lang_model.get_word_index(w))
            else:
                tokens.append(self.lang_model["UNK"])
        tokens = np.asarray(tokens, dtype=int)
        
        # Get the caption
        unique_tokens, unique_index = np.unique(tokens, return_index=True)
        unique_index = np.sort(unique_index)
        caption = " ".join([words[i] for i in unique_index])
        return caption, tokens
    
    def read_one_t2m_data(self, data):
        body = data["body"]
        text_list = data["text"]
        text_data = random.choice(text_list)    # Random sample one text
        caption_list = [t["caption"] for t in text_list]
        caption = text_data["caption"]
        t_tokens = text_data["tokens"]
        mot_len = body.shape[0]
        if mot_len > self.opt["window_size"]["t2m"]:
            # If motion sequence is longer than target lenght, we randomly select N frames from it. 
            # This approach partially acts as data augmentation during training.
            # indices = np.random.choice(mot_len, (self.opt["window_size"][key], ), replace=False)
            i = np.random.randint(0, mot_len - self.opt["window_size"]["t2m"])
            j = i + self.opt["window_size"]["t2m"]
            # indices = np.sort(indices)
            body = body[i:j]
            mot_len = body.shape[0]
        elif mot_len < self.opt["window_size"]["t2m"]:
            # If the motion sequence is shorter than target length, we pad zero poses.
            pad_len = self.opt["window_size"]["t2m"] - mot_len
            pad_pose = np.zeros((pad_len, body.shape[1]))
            body = np.concatenate([body, pad_pose], axis=0)
                
        batch = {"task": "t2m", "body": body, "text": caption, "text_list": caption_list, "length": mot_len}
        return batch

    def read_one_a2m_data(self, data):
        
        name = data["name"]
        body = data["body"] # The motion has already been downsampled to fps=30
        audio = data["audio"] # The fps of audio is 60 by default, and we don't modify it.
        duration = data["duration"]
        mot_len = body.shape[0]
        batch = {
            "body": body, "audio": audio, 
            "length": mot_len, "name": name, 
            "task": "a2m"
        }
        return batch
    
    def read_one_s2m_data(self, data):
        name = data["name"]
        audio = data["audio"]
        body = data["body"]
        left = data["left"]
        right = data["right"]
        caption, words = self.parse_word_tokens(data["word"])
        semantic = data["semantic"]
        emotion = self.parse_emotion_label(data["emotion"])
        speaker_id = np.array(data["speaker_id"], dtype=int)
        
        batch = {
            "body": body, "left": left, "right": right, 
            "audio": audio, "word": words, "caption": caption, 
            "semantic": semantic, "emotion": emotion, 
            "speaker_id": speaker_id, "name": name, "task": "s2m", "length": body.shape[0]
        }
        return batch
    
DATASET_MAP = {
    "HumanML3DTokenizerDataset": HumanML3DTokenizerDataset, 
    "AISTPPTokenizerDataset": AISTPPTokenizerDataset, 
    "BEATTokenizerDataset": BEATTokenizerDataset, 
    "UDEDatasetEval": UDEDatasetEval, 
    "UDEDataset": UDEDataset
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
    with open("configs/ude/config_ude_exp6.yaml", "r") as f:
        conf = yaml.safe_load(f)
    
    loader, _ = get_dataloader(data_conf=conf["data"]["dataset"], loader_conf=conf["data"]["loader"]["train"])
    for _, batch in enumerate(loader):
        pass
        # print(batch["t2m"]["body"].shape)