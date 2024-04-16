import os
from collections import defaultdict
from packaging import version
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 
    Wav2Vec2FeatureExtractor, 
    HubertModel
)
from transformers import __version__ as trans_version

class Wav2Vec2(nn.Module):
    def __init__(self, conf):
        super(Wav2Vec2, self).__init__()
        self.conf = conf
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.conf = conf
        for key in conf.keys():
            if "cache_dir" not in key: continue
            if not os.path.exists(conf[key]): os.makedirs(conf[key])
        print("Wav2Vec2 building from pretrained...")
        # self.model = Wav2Vec2ForCTC.from_pretrained(conf["model_name"], 
        #                                             cache_dir=conf["model_cache_dir"])
        self.model = Wav2Vec2ForCTC.from_pretrained(
            pretrained_model_name_or_path=conf["model_cache_dir"])
        print("Processor building from pretrained...")
        # self.processor = Wav2Vec2Processor.from_pretrained(conf["model_name"], 
        #                                                    cache_dir=conf["preprocessor_cache_dir"])
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
            
        
if __name__ == "__main__":
    import os, sys, librosa, wave
    sys.path.append(os.getcwd())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    wav_path = "../HumanPoseEstimation/GLAMR/assets/dynamic/s2m/clips/raw_video_1/000.wav"
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
        # "model_name": "ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt", 
        # "model_cache_dir": "networks/ude_v2/pretrained-model/ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt/model/", 
        # "preprocessor_cache_dir": "networks/ude_v2/pretrained-model/ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt/preprocessor/", 
        "model_name": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn", 
        "model_cache_dir": "networks/ude_v2/pretrained-model/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn/model/", 
        "preprocessor_cache_dir": "networks/ude_v2/pretrained-model/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn/preprocessor/", 
        "print": False, 
    }
    model = Wav2Vec2(conf=conf)    
    model.to(device)
    model.eval()    
    embeds, mask, transcription = model(audio, mask=None, decode_transcription=True)
    print(transcription)
    print(embeds.shape)
            
            