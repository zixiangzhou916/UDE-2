import os
from collections import defaultdict
from packaging import version
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2FeatureExtractor, 
    HubertForCTC, 
    HubertModel
)
from transformers import __version__ as trans_version

class HuBERT(nn.Module):
    def __init__(self, conf):
        super(HuBERT, self).__init__()
        self.conf = conf
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.conf = conf
        for key in conf.keys():
            if "cache_dir" not in key: continue
            if not os.path.exists(conf[key]): os.makedirs(conf[key])
        print("HuBERT building from pretrained...")
        # self.model = HubertForCTC.from_pretrained(conf["model_name"], 
        #                                          cache_dir=conf["model_cache_dir"])
        self.model = HubertForCTC.from_pretrained(
            pretrained_model_name_or_path=conf["model_cache_dir"])
        print("Processor building from pretrained...")
        # self.processor = Wav2Vec2Processor.from_pretrained(conf["model_name"], 
        #                                                cache_dir=conf["preprocessor_cache_dir"])
        self.processor = Wav2Vec2Processor.from_pretrained(
            pretrained_model_name_or_path=conf["preprocessor_cache_dir"])
        self.model.eval()
        self.model = self.model.to(self.device)
        
    def eval(self):
        self.model.eval()
        
    def to(self, device):
        self.model = self.model.to(device)
        
    def freeze(self):
        """Freeze the parameters to make them untrainable.
        """
        for p in self.model.parameters():
            p.requires_grad = False
            
    def forward(self, audios, mask=None, decode_transcription=False):
        """
        :param audios: [batch_size, seq_len]
        """
        with torch.no_grad():
            input_values = []
            attention_mask = []
            for audio in audios:
                processed = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                input_values.append(processed.input_values.to(self.device))
                attention_mask.append(processed.attention_mask.to(self.device))
            input_values = torch.cat(input_values, dim=0)
            attention_mask = torch.cat(attention_mask, dim=0)
            # print(input_values.shape, "|", attention_mask.shape)
            outputs = self.model(input_values, attention_mask=attention_mask, output_hidden_states=True)
            embeds = outputs.hidden_states
            attn = outputs.attentions
            if decode_transcription:
                logits = outputs.logits
                # print(embeds[-1].shape, "|", len(embeds))
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)
            mask = torch.ones(embeds[-1].size(0), embeds[-1].size(1)).bool().to(audios.device)
        
        if not decode_transcription:
            return embeds[-1], mask
        else:
            return embeds[-1], mask, transcription
        
    def encode_speech(self, audios, mask=None):
        with torch.no_grad():
            input_values = []
            attention_mask = []
            for audio in audios:
                processed = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                input_values.append(processed.input_values.to(self.device))
                attention_mask.append(processed.attention_mask.to(self.device))
            input_values = torch.cat(input_values, dim=0)
            attention_mask = torch.cat(attention_mask, dim=0)
            # print(input_values.shape, "|", attention_mask.shape)
            outputs = self.model(input_values, attention_mask=attention_mask, output_hidden_states=True)
            embeds = outputs.hidden_states
        return embeds[-1].mean(dim=1)
       
class HuBERT_Chinese(nn.Module):
    def __init__(self, conf):
        super(HuBERT_Chinese, self).__init__()
        self.conf = conf
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.conf = conf
        for key in conf.keys():
            if "cache_dir" not in key: continue
            if not os.path.exists(conf[key]): os.makedirs(conf[key])
        print("HuBERT building from pretrained...")
        self.model = HubertModel.from_pretrained(conf["model_name"], 
                                                 cache_dir=conf["model_cache_dir"])
        print("Processor building from pretrained...")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(conf["model_name"], 
                                                                  cache_dir=conf["preprocessor_cache_dir"])
        self.model.eval()
        self.model = self.model.to(self.device)
        
    def eval(self):
        self.model.eval()
        
    def to(self, device):
        self.model = self.model.to(device)
        
    def freeze(self):
        """Freeze the parameters to make them untrainable.
        """
        for p in self.model.parameters():
            p.requires_grad = False
            
    def forward(self, audios, mask=None, decode_transcription=False):
        """
        :param audios: [batch_size, seq_len]
        """
        with torch.no_grad():
            input_values = []
            attention_mask = []
            for audio in audios:
                processed = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                input_values.append(processed.input_values.to(self.device))
                attention_mask.append(processed.attention_mask.to(self.device))
            input_values = torch.cat(input_values, dim=0)
            attention_mask = torch.cat(attention_mask, dim=0)
            # print(input_values.shape, "|", attention_mask.shape)
            outputs = self.model(input_values, attention_mask=attention_mask, output_hidden_states=True)
            embeds = outputs.hidden_states
            attn = outputs.attentions
            mask = torch.ones(embeds[-1].size(0), embeds[-1].size(1)).bool().to(audios.device)
        
        return embeds[-1], mask
        
if __name__ == "__main__":
    import os, sys, librosa, wave
    sys.path.append(os.getcwd())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # wav_path = "../dataset/BEAT_v0.2.1/beat_english_v0.2.1/2/2_scott_1_8_8.wav"
    wav_path = "/mnt/user/zhouzixiang/projects/workspace/HumanPoseEstimation/GLAMR/assets/dynamic/s2m/clips/raw_video_1/000.wav"
    audio, sr = librosa.load(wav_path, sr=16000, mono=True)
    with wave.open(wav_path) as w:
        duration = w.getnframes() / w.getframerate()
    
    window_size = 160
    input_size = int(window_size * 16000 / 30)
    audios = [audio[i:i+input_size] for i in range(0, audio.shape[0]-input_size, input_size)]
    audios = np.stack(audios[:8], axis=0)
    audio = torch.from_numpy(audios).float().to(device)
    print('---', audio.shape)
    
    conf = {
        "model_name": "facebook/hubert-large-ls960-ft", 
        "model_cache_dir": "networks/ude_v2/pretrained-model/hubert-large-ls960-ft/model/", 
        "preprocessor_cache_dir": "networks/ude_v2/pretrained-model/hubert-large-ls960-ft/preprocessor/", 
        # "model_name": "TencentGameMate/chinese-hubert-large", 
        # "model_cache_dir": "networks/ude_v2/pretrained-model/TencentGameMate/chinese-hubert-large/model/", 
        # "preprocessor_cache_dir": "networks/ude_v2/pretrained-model/TencentGameMate/chinese-hubert-large/preprocessor/", 
        "print": False, 
    }
    model = HuBERT(conf=conf)
    model.to(device)
    model.eval()    
    embeds, mask = model(audio, mask=None)
    print(embeds[0])
    print(embeds.shape)
        
    # https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn