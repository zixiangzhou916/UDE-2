import os
from collections import defaultdict
from packaging import version
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertTokenizer, 
    BertModel
)
from transformers import __version__ as trans_version

class BERT(nn.Module):
    def __init__(self, conf):
        super(BERT, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.conf = conf
        for key in conf.keys():
            if "cache_dir" not in key: continue
            if not os.path.exists(conf[key]): os.makedirs(conf[key])
        print("Tokenizer building from pretrained...")
        # self.tokenizer = BertTokenizer.from_pretrained(conf["model_name"], 
        #                                                cache_dir=conf["tokenizer_cache_dir"])
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=conf["tokenizer_cache_dir"])
        print("BERT building from pretrained...")
        # self.model = BertModel.from_pretrained(conf["model_name"], 
        #                                        cache_dir=conf["model_cache_dir"])
        self.model = BertModel.from_pretrained(
            pretrained_model_name_or_path=conf["model_cache_dir"])

        self.model.eval()
        # self.model = self.model.to(self.device)
        
    def eval(self):
        self.model.eval()
        
    def to(self, device):
        self.model = self.model.to(device)
        
    def freeze(self):
        """Freeze the parameters to make them untrainable.
        """
        for p in self.model.parameters():
            p.requires_grad = False
      
    def forward(self, all_sens, mask=None, all_layers=False, output_hidden_layers=True, return_pooler=False):
        if "padding_len" in self.conf:
            padding_len = self.conf["padding_len"]
        else:
            padding_len = 128
        with torch.no_grad():
            # Tokenize input texts
            encoded_inputs = self.tokenizer(all_sens, return_tensors="pt", 
                                            padding=True, truncation=True)
            # print(encoded_inputs["input_ids"].shape)
            # Put tokens to current device
            encoded_inputs = {key: val.to(self.device) for key, val in encoded_inputs.items()}

            t = encoded_inputs["input_ids"].size(1)
            if t < padding_len: # Maximum token length is padding_len
                ids_pad = torch.zeros(len(all_sens), padding_len-t).long().to(self.device)
                types_pad = torch.zeros(len(all_sens), padding_len-t).long().to(self.device)
                masks_pad = torch.zeros(len(all_sens), padding_len-t).long().to(self.device)
                encoded_inputs["input_ids"] = torch.cat([encoded_inputs["input_ids"], ids_pad], dim=1)
                encoded_inputs["token_type_ids"] = torch.cat([encoded_inputs["token_type_ids"], types_pad], dim=1)
                encoded_inputs["attention_mask"] = torch.cat([encoded_inputs["attention_mask"], masks_pad], dim=1)
            else:
                encoded_inputs["input_ids"] = encoded_inputs["input_ids"][:, :padding_len]
                encoded_inputs["token_type_ids"] = encoded_inputs["token_type_ids"][:, :padding_len]
                encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"][:, :padding_len]
            # Extract features from token sequence
            output = self.model(**encoded_inputs)
            # Get last hidden state
            embeds = output.last_hidden_state
            pooler_output = output.pooler_output
            # Mask the padded parts
            embeds *= encoded_inputs["attention_mask"].float().unsqueeze(dim=-1)
            # Get attention mask
            masks = encoded_inputs["attention_mask"].bool().unsqueeze(dim=1)
                        
        if not return_pooler:
            return embeds, masks   # [B, 1, T]
        else:
            return embeds, pooler_output, masks
    
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.getcwd())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    conf = {
        "model_name": "bert-base-uncased", 
        "conf_cache_dir": "networks/ude_v2/pretrained-model/bert-base-uncased/conf/", 
        "tokenizer_cache_dir": "networks/ude_v2/pretrained-model/bert-base-uncased/tokenizer/", 
        "model_cache_dir": "networks/ude_v2/pretrained-model/bert-base-uncased/model/", 
        "print": False, 
        "padding_len": 34
    }
    model = BERT(conf=conf)
    model.to(device)
    model.eval()
    
    for name, param in model.state_dict().items():
        print(name,"|", param.min().item(), "|", param.max().item())
    
    texts = [
        "a person starts walking forward and then begins to run.",
        "a person throwing something at shoulder height.",
        "someone scrolls from right to left and then stands",
        "a person walks forward at moderate pace.",
        "a person dances and then runs backwards and forwards.",
        "a person grabbed soemthing and put it somewhere",
        "person standing to the side and checking his or her watch.",
        "character is moving slowly before moving into a moderate jog.",
        "a person strides forward for a few steps, then slows and stops.",
        "swinging arm down then standing still.",
        "a person runs forward and jumps over something, then turns around and jumps back over it.",
        "walking forward then stopping.",
        "a person hops on their left foot then their right",
        "a man is is doing basketball drills.",
        "the person is doing a military crawl across the floor",
        "this is a stupid test, i just want to know what's the difference between RoBERTa and ROBERTA?", 
        # "a person runs forward and jumps over something, then turns around and jumps back over it. a person runs forward and jumps over something, then turns around and jumps back over it. a person runs forward and jumps over something, then turns around and jumps back over it, jumps back over it. a person runs forward and jumps over something, then turns around and jumps back over it, jumps back over it. a person runs forward and jumps over something, then turns around and jumps back over it.",
    ]

    debug_texts = [
        "a person walks backward slowly.", 
        "a person walks forwards slowly.", 
        "a person runs backward.", 
        "a person runs forward.", 
        "person is walking straight backward.", 
        "person is walking straight forward.", 
        "a person moves side to side in a zig-zag fashion backward.",
        "a person moves side to side in a zig-zag fashion forward.",
        "a person stands and then walks backward.",
        "a person stands and then walks forward.",
        "a person jogs backward, diagonally.",
        "a person jogs forward, diagonally.",
    ]

    embeds, glob_emb, masks = model(debug_texts, mask=None, return_pooler=True)
    similarity = F.cosine_similarity(glob_emb[:, None], glob_emb[None], dim=-1)
    np.savetxt("similarity.txt", similarity.data.cpu().numpy(), fmt="%05f")
    
    print(embeds[0, :, 0])
