import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from transformers import (
    BertTokenizer, 
    BertModel
)

class TextProjectorV1(nn.Module):
    def __init__(self, conf):
        super(TextProjectorV1, self).__init__()
        self.linear1 = nn.Linear(conf["d_input"], conf["d_model"])
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1 = nn.BatchNorm1d(conf["d_model"])
        self.linear2 = nn.Linear(conf["d_model"], conf["d_output"])
        
    def forward(self, x):
        """
        :param x: [batch_size, dim]
        """
        out = self.bn1(self.act1(self.linear1(x)))
        out = self.linear2(out)
        return out
    
class TextProjectorV2(nn.Module):
    def __init__(self, conf):
        super(TextProjectorV2, self).__init__()
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=conf["d_model"],
            nhead=conf["n_head"],
            dim_feedforward=conf["d_inner"],
            dropout=conf["dropout"],
            activation=conf["activation"])
        self.transformer = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer, 
            num_layers=conf["n_layer"])
        self.final = nn.Linear(conf["d_model"], conf["d_output"])
        
    def forward(self, emb, mask):
        """
        :param emb: [batch_size, seq_len, dim] float tensor.
        :param mask: [batch_size, seq_len] boolean tensor.
        """
        emb = emb.permute(1, 0, 2)  # [seq_len, batch_size, dim]
        out = self.transformer(emb, src_key_padding_mask=~mask)
        out = out.permute(1, 0, 2)  # [batch_size, seq_len, dim]
        out = out * mask.float().unsqueeze(dim=-1)
        out = out.sum(dim=1) / mask.float().sum(dim=1, keepdim=True)    # [batch_size, dim]
        return out
    
class Text2MotionAlignmentV1(nn.Module):
    def __init__(
        self, conf, 
        text_encoder: nn.Module = None, 
        motion_encoder: nn.Module = None, 
        motion_decoder: nn.Module = None,
    ):
        super(Text2MotionAlignmentV1, self).__init__()
        self.conf = conf
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Build Sentence-BERT
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=conf["bert"]["tokenizer"])
        self.sbert = BertModel.from_pretrained(pretrained_model_name_or_path=conf["bert"]["model"])
        print("Sentence-BERT built successfully")
        
        if text_encoder is None:
            self.text_encoder = importlib.import_module(
                conf["text_encoder"]["arch_path"], package="networks").__getattribute__(
                    conf["text_encoder"]["arch_name"])(conf["text_encoder"])
            print("Pretrained text encoder built successfully")
        else:
            self.text_encoder = text_encoder
        
        if motion_encoder is None:
            self.motion_encoder = importlib.import_module(
                conf["motion_encoder"]["arch_path"], package="networks").__getattribute__(
                    conf["motion_encoder"]["arch_name"])(**conf["motion_encoder"])
            checkpoint = torch.load(conf["motion_encoder"]["checkpoint"], map_location=torch.device("cpu"))["encoder"]
            self.motion_encoder.load_state_dict(checkpoint, strict=True)
            print("Pretrained motion encoder built successfully")
        else:
            self.motion_encoder = motion_encoder
        
        if motion_decoder is None:
            self.motion_decoder = importlib.import_module(
                conf["motion_decoder"]["arch_path"], package="networks").__getattribute__(
                    conf["motion_decoder"]["arch_name"])(**conf["motion_decoder"])
            checkpoint = torch.load(conf["motion_decoder"]["checkpoint"], map_location=torch.device("cpu"))["decoder"]
            self.motion_decoder.load_state_dict(checkpoint, strict=True)
            print("Pretrained motion decoder built successfully")
        else:
            self.motion_decoder = motion_decoder
                
        self.trainables = nn.ModuleDict()
        for name, model_conf in conf["trainable"].items():
            self.trainables[name] = importlib.import_module(
            model_conf["arch_path"], package="networks").__getattribute__(
                model_conf["arch_name"])(model_conf)
    
        self.freeze()
        self.defreeze()
        
    def train(self):
        self.text_encoder.eval()
        self.motion_encoder.train()
        self.motion_decoder.train()
        self.sbert.eval()
        for _, model in self.trainables.items():
            model.train()
        
    def eval(self):
        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.motion_decoder.eval()
        self.sbert.eval()
        for _, model in self.trainables.items():
            model.eval()
            
    def freeze(self):
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.sbert.parameters():
            p.requires_grad = False
            
    def defreeze(self):
        for p in self.motion_encoder.parameters():
            p.requires_grad = True
        for p in self.motion_decoder.parameters():
            p.requires_grad = True
        for p in self.trainables.parameters():
            p.requires_grad = True
    
    def state_dict(self):
        state_dict = {}
        for name, val in super().state_dict().items():
            if "trainables" in name or "motion_encoder" in name or "motion_decoder" in name:
                state_dict[name] = val
        return state_dict
    
    def load_state_dict(self, state_dict, strict: bool = True):
        trainable_state = {}
        motion_encoder_state = {}
        motion_decoder_state = {}
        for name, val in state_dict.items():
            if "trainables" in name:
                sub_name = ".".join(name.split(".")[1:])
                trainable_state[sub_name] = val
            if "motion_encoder" in name:
                sub_name = ".".join(name.split(".")[1:])
                motion_encoder_state[sub_name] = val
            if "motion_decoder" in name:
                sub_name = ".".join(name.split(".")[1:])
                motion_decoder_state[sub_name] = val
    
        try:
            self.trainables.load_state_dict(trainable_state, strict=strict)
            self.motion_encoder.load_state_dict(motion_encoder_state, strict=strict)
            self.motion_decoder.load_state_dict(motion_decoder_state, strict=strict)
        except:
            raise ValueError("trainable parameters loading failed!")
    
    def parameters(self):
        return super().parameters()
    
    def forward(self, motion, text, lengths, decode=True):
        # Encode text description
        with torch.no_grad():
            text_emb, clip_emb, text_mask = self.text_encoder(text, mask=None, return_pooler=True)
        
        if "TextProjectorV1" in self.conf["trainable"]["text"]["arch_name"]:
            t_emb = self.trainables["text"](clip_emb)               # [B, C]
        elif "TextProjectorV2" in self.conf["trainable"]["text"]["arch_name"]:
            t_emb = self.trainables["text"](text_emb, text_mask[:, 0].bool())    # [B, C]
        
        # Encode motion sequence
        mask = torch.ones(motion.size(0), motion.size(1)).to(self.device)
        for i, l in enumerate(lengths):
            mask[i, l:] = 0
        mot_emb = self.motion_encoder(motion, mask.bool()).loc[:, 0]    # [B, C]
            
        # Calculate the text similarity in batch
        text_similarity = self.calc_text_similarity(text=text)
        
        output = {
            "m_emb": mot_emb,                   # Before normalization
            "t_emb": t_emb,                     # Before normalization
            "clip_emb": clip_emb,               # Before normalization
            "t_similarity": text_similarity
        }
        
        # Decode motion from embedding if specified
        if decode:
            # 1. Decode motion from text embedding
            # lengths = [x.size(0) for x in motion]
            recon_text = self.motion_decoder(t_emb.unsqueeze(dim=1), lengths=lengths)
            output["t_recon"] = recon_text
            
            # 2. Decode motion from motion embedidng
            recon_motion = self.motion_decoder(mot_emb.unsqueeze(dim=1), lengths=lengths)
            output["m_recon"] = recon_motion
        return output
    
    def encode_text(self, text):
        # Encode text description
        with torch.no_grad():
            text_emb, clip_emb, text_mask = self.text_encoder(text, mask=None, return_pooler=True)
        
        if "TextProjectorV1" in self.conf["trainable"]["text"]["arch_name"]:
            t_emb = self.trainables["text"](clip_emb)               # [B, C]
        elif "TextProjectorV2" in self.conf["trainable"]["text"]["arch_name"]:
            t_emb = self.trainables["text"](text_emb, text_mask[:, 0].bool())    # [B, C]
        
        return t_emb
    
    def encode_motion(self, motion, lengths):
        # Encode motion sequence
        mask = torch.ones(motion.size(0), motion.size(1)).to(self.device)
        for i, l in enumerate(lengths):
            mask[i, l:] = 0
        mot_emb = self.motion_encoder(motion, mask.bool()).loc[:, 0]    # [B, C]
        return mot_emb
    
    @torch.no_grad()
    def calc_text_similarity(self, text):
        tokenization = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        model_output = self.sbert(
            input_ids=tokenization["input_ids"].to(self.device), 
            attention_mask=tokenization["attention_mask"].to(self.device), 
            token_type_ids=tokenization["token_type_ids"].to(self.device)
        )
        # Perform pooling
        token_embeddings = model_output[0]  # First element of model_output
        input_mask_expanded = tokenization['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float().to(self.device)
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)  # [B, C]
        # Calculate Similarity
        sentence_similarity = F.cosine_similarity(sentence_embeddings[:, None], sentence_embeddings[None], dim=-1)
        
        # mask = torch.eye(sentence_similarity.size(0)).float().to(self.device)
        # sentence_similarity = sentence_similarity * (1. - mask)
        return sentence_similarity
    
    @staticmethod
    def calc_infonce(text_embeddings, motion_embeddings, text_similarity, temperature=0.1):
        """
        :param text_embeddings: [batch_size, num_dim]
        :param motion_embeddings: [batch_size, num_dim]
        :param text_similarity: [batch_size, batch_size]
        """
        batch_size = text_embeddings.size(0)
        mask = text_similarity.lt(0.85).float()
        mask += torch.eye(batch_size).float().to(text_embeddings.device)
        similarity = F.cosine_similarity(text_embeddings, motion_embeddings, dim=-1)                         # [N]
        similarity_expanded = F.cosine_similarity(text_embeddings[:, None], motion_embeddings[None], dim=-1) # [N, N]
                
        exp_similarity = torch.exp(similarity / temperature)                    # [N]
        exp_similarity_expanded = torch.exp(similarity_expanded / temperature)  # [N, N]
        exp_similarity_expanded = exp_similarity_expanded * mask
        
        tex2mot = exp_similarity / exp_similarity_expanded.sum(dim=1)           # [N]
        mot2tex = exp_similarity / exp_similarity_expanded.sum(dim=0)           # [N]
        nce_loss = -1 / (2*batch_size) * (torch.log(tex2mot + 1e-09) + torch.log(mot2tex + 1e-09)).sum()
                
        return nce_loss
    
""" DEBUG """
class Text2MotionSyncV2(nn.Module):
    def __init__(
        self, conf, 
        text_encoder: nn.Module = None, 
        motion_encoder: nn.Module = None, 
        motion_decoder: nn.Module = None,
    ):
        super(Text2MotionSyncV2, self).__init__()
        self.conf = conf
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Build Sentence-BERT
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=conf["bert"]["tokenizer"])
        self.sbert = BertModel.from_pretrained(pretrained_model_name_or_path=conf["bert"]["model"])
        print("Sentence-BERT built successfully")
        
        if text_encoder is None:
            self.text_encoder = importlib.import_module(
                conf["text_encoder"]["arch_path"], package="networks").__getattribute__(
                    conf["text_encoder"]["arch_name"])(conf["text_encoder"])
            print("Pretrained text encoder built successfully")
        else:
            self.text_encoder = text_encoder
        
        if motion_encoder is None:
            self.motion_encoder = importlib.import_module(
                conf["motion_encoder"]["arch_path"], package="networks").__getattribute__(
                    conf["motion_encoder"]["arch_name"])(**conf["motion_encoder"])
            checkpoint = torch.load(conf["motion_encoder"]["checkpoint"], map_location=torch.device("cpu"))["encoder"]
            self.motion_encoder.load_state_dict(checkpoint, strict=True)
            print("Pretrained motion encoder built successfully")
        else:
            self.motion_encoder = motion_encoder
        
        if motion_decoder is None:
            self.motion_decoder = importlib.import_module(
                conf["motion_decoder"]["arch_path"], package="networks").__getattribute__(
                    conf["motion_decoder"]["arch_name"])(**conf["motion_decoder"])
            checkpoint = torch.load(conf["motion_decoder"]["checkpoint"], map_location=torch.device("cpu"))["decoder"]
            self.motion_decoder.load_state_dict(checkpoint, strict=True)
            print("Pretrained motion decoder built successfully")
        else:
            self.motion_decoder = motion_decoder
                
        self.trainables = nn.ModuleDict()
        for name, model_conf in conf["trainable"].items():
            self.trainables[name] = importlib.import_module(
            model_conf["arch_path"], package="networks").__getattribute__(
                model_conf["arch_name"])(model_conf)
    
        self.freeze()
        
    def train(self):
        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.motion_decoder.eval()
        self.sbert.eval()
        for _, model in self.trainables.items():
            model.train()
        
    def eval(self):
        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.motion_decoder.eval()
        self.sbert.eval()
        for _, model in self.trainables.items():
            model.eval()
            
    def freeze(self):
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.motion_encoder.parameters():
            p.requires_grad = False
        for p in self.motion_decoder.parameters():
            p.requires_grad = False
        for p in self.sbert.parameters():
            p.requires_grad = False
            
    def state_dict(self):
        state_dict = {}
        for name, val in super().state_dict().items():
            if "trainables" in name:
                state_dict[name] = val
        return state_dict
    
    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict_reorg = {}
        for name, val in state_dict.items():
            if "trainables" in name:
                sub_name = ".".join(name.split(".")[1:])
                state_dict_reorg[sub_name] = val
    
        try:
            self.trainables.load_state_dict(state_dict_reorg, strict=strict)
        except:
            raise ValueError("trainable parameters loading failed!")
        
    def parameters(self):
        return super().parameters()
    
    def forward(self, motion, text, decode=True):
        # Encode text description
        with torch.no_grad():
            text_emb, clip_emb, text_mask = self.text_encoder(text, mask=None, return_pooler=True)
        
        if "TextProjectorV1" in self.conf["trainable"]["text"]["arch_name"]:
            t_emb = self.trainables["text"](clip_emb)               # [B, C]
        elif "TextProjectorV2" in self.conf["trainable"]["text"]["arch_name"]:
            t_emb = self.trainables["text"](text_emb, text_mask[:, 0].bool())    # [B, C]
        
        # Encode motion sequence
        with torch.no_grad():
            # motion = motion.permute(0, 2, 1)
            mot_emb = self.motion_encoder(motion).loc[:, 0]    # [B, C]
            
        # Calculate the text similarity in batch
        text_similarity = self.calc_text_similarity(text=text)
        
        output = {
            "m_emb": mot_emb,                   # Before normalization
            "t_emb": t_emb,                     # Before normalization
            "clip_emb": clip_emb,               # Before normalization
            "t_similarity": text_similarity
        }
        
        # Decode motion from embedding if specified
        if decode:
            # 1. Decode motion from text embedding
            lengths = [x.size(0) for x in motion]
            recon_text = self.motion_decoder(t_emb.unsqueeze(dim=1), lengths=lengths)
            output["t_recon"] = recon_text
            
            # 2. Decode motion from motion embedidng
            recon_motion = self.motion_decoder(mot_emb.unsqueeze(dim=1), lengths=lengths)
            output["m_recon"] = recon_motion
        return output
    
    def encode_text(self, text):
        # Encode text description
        with torch.no_grad():
            text_emb, clip_emb, text_mask = self.text_encoder(text, mask=None, return_pooler=True)
        
        if "TextProjectorV1" in self.conf["trainable"]["text"]["arch_name"]:
            t_emb = self.trainables["text"](clip_emb)               # [B, C]
        elif "TextProjectorV2" in self.conf["trainable"]["text"]["arch_name"]:
            t_emb = self.trainables["text"](text_emb, text_mask[:, 0].bool())    # [B, C]
        
        return t_emb
    
    def encode_motion(self, motion, lengths):
        # Encode motion sequence
        with torch.no_grad():
            # motion = motion.permute(0, 2, 1)
            mot_emb = self.motion_encoder(motion).loc[:, 0]    # [B, C]
        
        return mot_emb
    
    @torch.no_grad()
    def calc_text_similarity(self, text):
        tokenization = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        model_output = self.sbert(
            input_ids=tokenization["input_ids"].to(self.device), 
            attention_mask=tokenization["attention_mask"].to(self.device), 
            token_type_ids=tokenization["token_type_ids"].to(self.device)
        )
        # Perform pooling
        token_embeddings = model_output[0]  # First element of model_output
        input_mask_expanded = tokenization['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float().to(self.device)
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)  # [B, C]
        # Calculate Similarity
        sentence_similarity = F.cosine_similarity(sentence_embeddings[:, None], sentence_embeddings[None], dim=-1)
        
        # mask = torch.eye(sentence_similarity.size(0)).float().to(self.device)
        # sentence_similarity = sentence_similarity * (1. - mask)
        return sentence_similarity
    
    @staticmethod
    def calc_infonce(text_embeddings, motion_embeddings, text_similarity, temperature=0.1):
        """
        :param text_embeddings: [batch_size, num_dim]
        :param motion_embeddings: [batch_size, num_dim]
        :param text_similarity: [batch_size, batch_size]
        """
        batch_size = text_embeddings.size(0)
        mask = text_similarity.lt(0.85).float()
        mask += torch.eye(batch_size).float().to(text_embeddings.device)
        similarity = F.cosine_similarity(text_embeddings, motion_embeddings, dim=-1)                         # [N]
        similarity_expanded = F.cosine_similarity(text_embeddings[:, None], motion_embeddings[None], dim=-1) # [N, N]
        
        exp_similarity = torch.exp(similarity / temperature)                    # [N]
        exp_similarity_expanded = torch.exp(similarity_expanded / temperature)  # [N, N]
        exp_similarity_expanded = exp_similarity_expanded * mask
        
        tex2mot = exp_similarity / exp_similarity_expanded.sum(dim=1)           # [N]
        mot2tex = exp_similarity / exp_similarity_expanded.sum(dim=0)           # [N]
        nce_loss = -1 / (2*batch_size) * (torch.log(tex2mot + 1e-09) + torch.log(mot2tex + 1e-09)).sum()
                
        return nce_loss
    