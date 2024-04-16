import os, sys
sys.path.append(os.getcwd())
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.ude.layers import *
import numpy as np
import random
import math
import copy

def get_pad_mask(batch_size, seq_len, non_pad_lens):
    non_pad_lens = non_pad_lens.data.tolist()
    mask_2d = torch.zeros((batch_size, seq_len), dtype=torch.float32)
    for i, cap_len in enumerate(non_pad_lens):
        mask_2d[i, :cap_len] = 1
    return mask_2d.unsqueeze(1).bool()

def get_pad_mask_idx(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1)

def get_subsequent_mask(seq):
    sz_b, seq_len = seq.shape
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, seq_len, seq_len)), diagonal=1)).bool()
    return subsequent_mask.to(seq.device)

def get_clip_textencoder_mask(tokens):
    """
    :param tokens: [batch_size, 77]
    """
    mask = torch.zeros_like(tokens)
    for i in range(tokens.shape[0]):
        id = tokens[i].argmax(-1)
        mask[i, :id+1] = 1
    mask = mask.bool()
    return mask.unsqueeze(1)

def top_k_logits(logits, k):
    """
    :param logits: [num_seq, num_dim]
    """
    dim = logits.dim()
    if dim == 3:
        logits = logits.squeeze(dim=1)
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    if dim == 3:
        out = out.unsqueeze(dim=1)
    return out

def minimize_special_token_logits(logits, k):
    """
    :param logits: [num_seq, num_dim]
    """
    dim = logits.dim()
    if dim == 3:
        logits = logits.squeeze(dim=1)
    out = logits.clone()
    out[..., :k] = -float('Inf')
    if dim == 3:
        out = out.unsqueeze(dim=1)
    return out

def check_nan(tensor):
    has_nan = torch.isnan(tensor)
    if has_nan.float().sum() > 0: 
        print(tensor)

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
"""UDE model, the model architecture follows UDE: https://github.com/zixiangzhou916/UDE/tree/master
1. We unify the text-to-motion, audio(music)-to-motion, speech-to-motion tasks in one pipeline.
"""
class UDEModel(nn.Module):
    def __init__(self, conf):
        super(UDEModel, self).__init__()
        self.conf = conf
        self.sos_id = 0
        self.eos_id = 1
        self.pad_id = 2
        self.special_token_embedding = nn.Embedding(num_embeddings=3, embedding_dim=conf.get("d_embed", 512))

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.style_proj = nn.Linear(conf["decoder"]["d_model"], conf["decoder"]["d_latent"])

        layers = []
        for _ in range(conf["decoder"]["n_mlp"]):
            layers.append(
                nn.Linear(in_features=conf["decoder"]["d_latent"], out_features=conf["decoder"]["d_latent"])
            )
        self.style = nn.Sequential(*layers)

        # Encoder
        self.encoder = importlib.import_module(
            conf["encoder"]["arch_path"], package="networks").__getattribute__(
                conf["encoder"]["arch_name"])(**conf["encoder"])
        
        # Decoder
        self.decoder = importlib.import_module(
            conf["gpt"]["arch_path"], package="networks").__getattribute__(
                conf["gpt"]["arch_name"])(conf["gpt"])

        # Pretrained model
        if "pretrained" in conf.keys():
            self.pretrained_models = nn.ModuleDict()
            for key, item in conf["pretrained"].items():
                self.pretrained_models[key] = importlib.import_module(
                    item["arch_path"], package="networks").__getattribute__(
                        item["arch_name"])(item)
                self.pretrained_models[key].eval()
                self.pretrained_models[key].to(self.device)
                self.pretrained_models[key].freeze()    # Freeze the parameters, we don't train them
                print("Pretrained {:s} encoder loaded".format(key))
        
        # Quantizers
        self.quantizers = {}

    def train(self):
        self.style_proj.train()
        self.style.train()
        self.encoder.train()
        self.decoder.train()
        self.special_token_embedding.train()
        for key in self.conf["trainable"]:
            self.pretrained_models[key].train()
        
    def eval(self):
        self.style_proj.eval()
        self.style.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.special_token_embedding.eval()
        for key in self.conf["trainable"]:
            self.pretrained_models[key].eval()
    
    def load_state_dict(self, state_dict, strict: bool = True):
        # Re-organize the state_dict
        state_dict_keys = ["encoder", 
                           "decoder", 
                           "style_proj", 
                           "special_token_embedding", 
                           "style"] + ["pretrained_models.{:s}".format(key) 
                                       for key in self.conf["trainable"]]
        state_dict_reorg = {key: {} for key in state_dict_keys}
        for name, val in state_dict.items():
            key = name.split(".")[0]
            sub_name = ".".join(name.split(".")[1:])
            if key == "pretrained_models":
                key = ".".join(name.split(".")[:2])
                sub_name = ".".join(name.split(".")[2:])
            if key in state_dict_reorg.keys():
                state_dict_reorg[key][sub_name] = val
            else:
                state_dict_reorg[key] = {sub_name: val}
    
        try:
            self.encoder.load_state_dict(state_dict_reorg["encoder"], strict=strict)
        except:
            raise ValueError("encoder parameters loading failed!")
        try:
            self.decoder.load_state_dict(state_dict_reorg["decoder"], strict=strict)
        except:
            raise ValueError("decoder parameters loading failed!")
        try:
            self.style_proj.load_state_dict(state_dict_reorg["style_proj"], strict=strict)
        except:
            raise ValueError("style_proj parameters loading failed!")
        try:
            self.style.load_state_dict(state_dict_reorg["style"], strict=strict)
        except:
            raise ValueError("style parameters loading failed!")
        try:
            self.special_token_embedding.load_state_dict(state_dict_reorg["special_token_embedding"], strict=strict)
        except:
            raise ValueError("special_token_embedding parameters loading failed!")
        
        for key in self.conf["trainable"]:
            try:
                self.pretrained_models[key].load_state_dict(state_dict_reorg["pretrained_models.{:s}".format(key)], strict=strict)
            except:
                raise ValueError("{:s} parameters loading failed!".format(key))

    def state_dict(self):
        state_dict_keys = ["encoder", 
                           "decoder", 
                           "style_proj", 
                           "special_token_embedding", 
                           "style"] + ["pretrained_models.{:s}".format(key) 
                                       for key in self.conf["trainable"]]
        state_dict = {}
        for name, val in super().state_dict().items():
            key = name.split(".")[0]
            if key == "pretrained_models":
                key = ".".join(name.split(".")[:2])
            if key in state_dict_keys:
                state_dict[name] = val
        return state_dict
    
    def print(self):
        for name, val in super().state_dict().items():
            print(name, val.min().item(), val.max().item())

    def setup_quantizer(
        self, quantizer, name
    ):
        self.quantizers[name] = copy.deepcopy(quantizer)
        for p in self.quantizers[name].parameters():
            p.requires_grad = False
    
    def setup_motion_encoder(self, encoder):
        self.motion_encoder = copy.deepcopy(encoder)
        for p in self.motion_encoder.parameters():
            p.requires_grad = False
        self.motion_encoder.eval()
            
    def tokens_to_embeddings(self, tokens, name):
        """
        :param tokens: [batch_size, seq_len]
        """
        sos_id = torch.tensor(self.sos_id).long().to(self.device)
        sos_embeds = self.special_token_embedding(sos_id).view(1, -1)
        eos_id = torch.tensor(self.eos_id).long().to(self.device)
        eos_embeds = self.special_token_embedding(eos_id).view(1, -1)
        pad_id = torch.tensor(self.pad_id).long().to(self.device)
        pad_embeds = self.special_token_embedding(pad_id).view(1, -1)
        
        embeds = []
        for tok in tokens:
            mask = tok.gt(self.pad_id)
            valid_tok = tok[mask]
            valid_embeds = self.quantizers[name].get_codebook_entry(valid_tok-3)
            valid_embeds = torch.cat([sos_embeds, valid_embeds, eos_embeds], dim=0)
            pad_len = tokens.size(1) - valid_embeds.size(0)
            if pad_len > 0:
                valid_embeds = torch.cat([valid_embeds, pad_embeds.repeat(pad_len, 1)], dim=0)
            embeds.append(valid_embeds)
        embeds = torch.stack(embeds, dim=0)
        return embeds
    
    def text_to_motion(
        self, text, input_ids
    ):
        # Get input embeds
        input_embeds = self.tokens_to_embeddings(tokens=input_ids, name="t2m_body")
        # Encode text input
        text_emb, clip_emb, text_mask = self.pretrained_models["text_encoder"](
            text, mask=None, return_pooler=True)

        # Encode setence embedding and word embedding
        glob_emb, seq_emb, cond_mask = self.encoder.encode_text(
            t_seq=text_emb, t_mask=text_mask[:, 0])

        latent = self.style_proj(glob_emb)            
        cond_emb = torch.cat((latent, seq_emb), dim=1)
        cond_length = cond_emb.size(1)

        # Decode using GPT architecture
        out = self.decoder(input_embeds, cond=cond_emb, cond_mask=cond_mask, type="t2m", part="body")
        
        cond_embs = {
            "pretrained_emb": clip_emb, 
            "fused_emb": glob_emb.squeeze(dim=1), 
            "fused_seq_emb": seq_emb, 
            "attn_mask": cond_mask.float(), 
            "text": text, 
        }

        return {"body": out}, glob_emb, seq_emb, cond_embs

    def audio_to_motion(
        self, audio, input_ids
    ):
        # Get input embeds
        input_embeds = self.tokens_to_embeddings(tokens=input_ids, name="a2m_body")
        # Encode the audio and motion primitives as cross-modality embedding
        a_mask = torch.ones(audio.size(0), audio.size(1)).bool().to(self.device)
        # Encode the audio sequence
        audio_emb, a_mask = self.pretrained_models["audio_encoder"](audio, mask=a_mask)
                
        glob_emb, seq_emb, cond_mask = self.encoder.encode_audio(
            a_seq=audio_emb, a_mask=a_mask)
        
        latent = self.style_proj(glob_emb)            
        cond_emb = torch.cat((latent, seq_emb), dim=1)
        cond_length = cond_emb.size(1)
        
        # Decode using GPT architecture
        out = self.decoder(input_embeds, cond=cond_emb, cond_mask=cond_mask, type="a2m", part="body")
        
        cond_embs = {
            "pretrained_emb": audio_emb.mean(dim=1), 
            "fused_emb": glob_emb.squeeze(dim=1), 
            "fused_seq_emb": seq_emb, 
            "attn_mask": cond_mask.float(), 
        }
        
        return {"body": out}, glob_emb, seq_emb, cond_embs
    
    def speech_to_motion(
        self, audio, ids, emotion, input_ids_dict
    ):
        """Speech to motion. 
        We consider audio sequence and the associated word tokens sequence as condition. 
        We don't encode speech text as condition.
        """
        # Get input embeds
        input_embeds_dict = {
            key: self.tokens_to_embeddings(tokens=val, name="s2m_{:s}".format(key)) 
            for key, val in input_ids_dict.items()
        }
        # Encode speech audio
        speech_emb, speech_mask = self.pretrained_models["speech_encoder"](audio, mask=None)
        
        glob_emb, seq_emb, cond_mask = self.encoder.encode_speech(
            s_seq=speech_emb, emo=emotion, ids=ids, s_mask=speech_mask)
        
        latent = self.style_proj(glob_emb)
        cond_emb = torch.cat((latent, seq_emb), dim=1)
        cond_length = cond_emb.size(1)
        
        # Decode using GPT architecture
        output = {}
        for part, input_embeds in input_embeds_dict.items():
            out = self.decoder(input_embeds, cond=cond_emb, cond_mask=cond_mask, type="s2m", part=part)
            output[part] = out
        
        cond_embs = {
            "pretrained_emb": speech_emb.mean(dim=1), 
            "fused_emb": glob_emb.squeeze(dim=1), 
            "fused_seq_emb": seq_emb, 
            "attn_mask": cond_mask.float(), 
        }
        
        return output, glob_emb, seq_emb, cond_embs
    
    def encode_text(self, text, task="text", vq_task="text"):
        """
        Encode the text embeddings using CLIP Text-Encoder and our fusion encoder, respectively.
        """            
        with torch.no_grad():
            # Encode text input
            text_emb, clip_emb, text_mask = self.pretrained_models["text_encoder"](text, mask=None, return_pooler=True)
            # text_emb, text_mask = self.pretrained_models["text_encoder"](text, mask=None, return_pooler=False)
            
            # Encode setence embedding and word embedding
            glob_emb, seq_emb, cond_mask = self.encoder.encode_text(
                t_seq=text_emb, t_mask=text_mask[:, 0])

        return text_emb, clip_emb, seq_emb, glob_emb[:, 0], cond_mask
    
    def encode_audio(self, audio, task="audio", vq_task="audio"):
        with torch.no_grad():
            # Encode the audio and motion primitives as cross-modality embedding
            a_mask = torch.ones(audio.size(0), audio.size(1)).bool().to(self.device)

            audio_emb, a_mask = self.pretrained_models["audio_encoder"](audio, mask=a_mask)

            glob_emb, seq_emb = self.encoder.encode_audio(
                a_seq=audio_emb, a_mask=a_mask)
        return audio_emb.mean(dim=1), seq_emb, glob_emb[:, 0]
    
    def encode_speech(self, speech, ids, emotion, task="speech", vq_task="speech"):
        with torch.no_grad():
            # Encode speech audio
            speech_emb, speech_mask = self.pretrained_models["speech_encoder"](speech, mask=None)
            
            glob_emb, seq_emb, cond_mask = self.encoder.encode_speech(
                s_seq=speech_emb, 
                emo=emotion, 
                ids=ids, 
                s_mask=speech_mask)
            
        return speech_emb.mean(dim=1), seq_emb, glob_emb[:, 0]
    
    @torch.no_grad()
    def get_motion_embedding(self, motion, lengths):
        """
        :param motion: [batch_size, seq_len, num_dim]
        :param lengths: [batch_size]
        """
        mask = torch.ones(motion.size(0), motion.size(1)).to(self.device)
        for i, l in enumerate(lengths):
            mask[i, l:] = 0
                
        # Encode motion
        motion_embedding = self.motion_encoder(motion, mask.bool()).loc
        return motion_embedding
    
    @torch.no_grad()
    def generate_text_to_motion(
        self, text, max_num_tokens=50, 
        topk=1, sas=False, temperature=1.0, 
        task="t2m", part="body"
    ):
        """
        :param text: list of textual description.
        :param max_num_tokens: maximum number of tokens allowed to be generated.
        :param topk: sample from top-k candidates.
        :param sas: use semantic-aware-sampling or not.
        :param temperature: controls the smoothness of semantic-aware distribution, large temperature smoothens the distribution
        """
        # Encode text input
        text_emb, clip_emb, text_mask = self.pretrained_models["text_encoder"](
            text, mask=None, return_pooler=True)

        # Encode setence embedding and word embedding
        glob_emb, seq_emb, cond_mask = self.encoder.encode_text(
            t_seq=text_emb, t_mask=text_mask[:, 0])
        latent = self.style_proj(glob_emb)            
        cond_emb = torch.cat((latent, seq_emb), dim=1)
        
        # Predict next token autoregressively
        sos_id = torch.tensor(self.sos_id).long().to(self.device)
        sos_embeds = self.special_token_embedding(sos_id).view(1, 1, -1)
        input_embeds = sos_embeds
        pred_tokens = []
        for _ in range(max_num_tokens):
            output = self.decoder(input_embeds, cond=cond_emb, cond_mask=cond_mask, type=task, part=part)
            raw_pred_logit = output[:, -1:] # [1, 1, C]
            if topk == 1:
                # Sample the token with highest probability
                pred_logit = F.softmax(raw_pred_logit.clone(), dim=-1)
                pred_token = pred_logit.argmax(dim=-1)
            else:
                pred_token = self.sample_one_token(
                    raw_pred_logit=raw_pred_logit.clone(), 
                    topk=topk, sas=sas, 
                    temperature=temperature, 
                    task=task, part=part)
            if pred_token.item() > self.pad_id:
                pred_tokens.append(pred_token)
                pred_emb = self.quantizers["{:s}_{:s}".format(task, part)].get_codebook_entry(pred_token-3).contiguous().view(1, 1, -1) # [1, 1, C]
                input_embeds = torch.cat([input_embeds, pred_emb], dim=1)
            else:
                if len(pred_tokens) == 0:
                    # We force the sample one motion token, although its probability is not highest.
                    pred_logit = minimize_special_token_logits(raw_pred_logit.clone(), k=3)
                    pred_token = self.sample_one_token(
                        raw_pred_logit=pred_logit.clone(), 
                        topk=topk, sas=sas, 
                        temperature=temperature, 
                        task=task, part=part)
                    pred_tokens.append(pred_token)
                    pred_emb = self.quantizers["{:s}_{:s}".format(task, part)].get_codebook_entry(pred_token-3).contiguous().view(1, 1, -1) # [1, 1, C]
                    input_embeds = torch.cat([input_embeds, pred_emb], dim=1)
                else:
                    break
        
        pred_tokens = torch.cat(pred_tokens, dim=1) # [1, T]
        return pred_tokens

    @torch.no_grad()
    def generate_audio_to_motion(
        self, audio, tokens=None, max_num_tokens=50, 
        topk=1, sas=False, 
        block_size=160, temperature=1.0, 
        task="a2m", part="body"
    ):
        """
        :param audio: music condition, [batch_size, music_seq_len]
        :param tokens: (optional)[batch_size, seq_len]
        :param max_num_tokens: maximum number of tokens allowed to be generated.
        :param topk: sample from top-k candidates.
        :param sas: use semantic-aware-sampling or not.
        :param temperature: controls the smoothness of semantic-aware distribution, large temperature smoothens the distribution
        """
        # Define some common parameters
        motion_block_size = block_size
        audio_block_size = int(motion_block_size * 16000 / 30)
        audio_step_size = int(16000 / 30)
        
        # Predict next token autoregressively
        if tokens is None:
            sos_id = torch.tensor(self.sos_id).long().to(self.device)
            input_embeds = self.special_token_embedding(sos_id).view(1, 1, -1)
            pred_tokens = []
        if tokens is not None:
            input_embeds = self.tokens_to_embeddings(tokens=tokens, name="{:s}_{:s}".format(task, part))
            input_embeds = input_embeds[:, :-1]
            pred_tokens = [tokens[:, 1:-1]]
        
        # We first generate the first block
        if tokens is None:
            primitive_token_size = 0
        else:
            primitive_token_size = tokens.size(1) - 1
        
        pred_token, input_embeds = self.generate_motion_tokens_from_audio(
            audio=audio[:, :audio_block_size], 
            input_embeds=input_embeds, 
            num_tokens=motion_block_size//4-primitive_token_size, 
            task="a2m", part=part, topk=topk, sas=sas, 
            temperature=temperature)
        pred_tokens.append(pred_token)
        
        # Keep the maximum length less than block size
        input_embeds = input_embeds[:, 1:]
        
        # Auto-regressively generate rest tokens
        for motion_index in range(1, max_num_tokens, 1):
            audio_index_i = int(motion_index * 16000 / 30)
            audio_index_j = audio_index_i + audio_block_size
            inp_audio_segment = audio[:, audio_index_i:audio_index_j]
            # Make sure the length of condition music piece equals audio_block_size
            if inp_audio_segment.size(-1) < audio_block_size:
                break
            pred_token, input_embeds = self.generate_motion_tokens_from_audio(
                audio=inp_audio_segment, 
                input_embeds=input_embeds, num_tokens=1, 
                task="a2m", part=part, topk=topk, sas=sas, 
                temperature=temperature)
            pred_tokens.append(pred_token)
            if input_embeds.size(1) > (motion_block_size // 4):
                input_embeds = input_embeds[:, 1:]  # We keep the maximum length of input_embeds fixed
        
        pred_tokens = torch.cat(pred_tokens, dim=1) # [1, T]
        return pred_tokens

    @torch.no_grad()
    def generate_speech_to_motion(
        self, audio, emotion, speaker_id, tokens=None, 
        max_num_tokens=50, 
        topk=1, sas=False, 
        block_size=160, temperature=1.0, 
        task="s2m", part="body"
    ):
        """
        :param audio: speech condition, [batch_size, music_seq_len]
        :param emotion: emotion ID, [batch_size]
        :param speaker_id: speaker ID, [batch_size]
        :param max_num_tokens: maximum number of tokens allowed to be generated.
        :param topk: sample from top-k candidates.
        :param sas: use semantic-aware-sampling or not.
        :param temperature: controls the smoothness of semantic-aware distribution, large temperature smoothens the distribution
        """
        # Define some common parameters
        motion_block_size = block_size
        audio_block_size = int(motion_block_size * 16000 / 30)
        audio_step_size = int(16000 / 30)
        
        # Predict next token autoregressively
        if tokens is None:
            sos_id = torch.tensor(self.sos_id).long().to(self.device)
            input_embeds = self.special_token_embedding(sos_id).view(1, 1, -1)
            pred_tokens = []
        else:
            input_embeds = self.tokens_to_embeddings(tokens=tokens, name="{:s}_{:s}".format(task, part))
            input_embeds = input_embeds[:, :-1]
            # pred_tokens = [tokens[:, 1:-1]]
            pred_tokens = [tokens[:, 1:]]
        
        # We first generate the first block
        if tokens is None:
            primitive_token_size = 0
        else:
            primitive_token_size = tokens.size(1) - 1
        
        pred_token, input_embeds = self.generate_motion_tokens_from_speech(
            audio=audio[:, :audio_block_size], 
            emotion=emotion, speaker_id=speaker_id, 
            input_embeds=input_embeds, 
            num_tokens=motion_block_size//4-primitive_token_size, 
            task="s2m", part=part, topk=topk, sas=sas, 
            temperature=temperature)
        pred_tokens.append(pred_token)
        
        # Keep the maximum length less than block size
        input_embeds = input_embeds[:, 1:]
        
        # Auto-regressively generate rest tokens
        for motion_index in range(1, max_num_tokens, 1):
            audio_index_i = int(motion_index * 16000 / 30)
            audio_index_j = audio_index_i + audio_block_size
            inp_audio_segment = audio[:, audio_index_i:audio_index_j]
            # Make sure the length of condition music piece equals audio_block_size
            if inp_audio_segment.size(-1) < audio_block_size:
                break
            pred_token, input_embeds = self.generate_motion_tokens_from_speech(
                audio=inp_audio_segment, 
                emotion=emotion, speaker_id=speaker_id, 
                input_embeds=input_embeds, num_tokens=1, 
                task="s2m", part=part, topk=topk, sas=sas, 
                temperature=temperature)
            pred_tokens.append(pred_token)
            
            if input_embeds.size(1) > (motion_block_size // 4):
                input_embeds = input_embeds[:, 1:]  # We keep the maximum length of input_embeds fixed
        
        pred_tokens = torch.cat(pred_tokens, dim=1) # [1, T]
        return pred_tokens
    
    @torch.no_grad()
    def generate_motion_tokens_from_audio(
        self, audio, input_embeds, 
        num_tokens=1, task="a2m", part="body", 
        topk=1, sas=False, temperature=1.0, 
    ):
        # Encode the audio and motion primitives as cross-modality embedding
        a_mask = torch.ones(audio.size(0), audio.size(1)).bool().to(self.device)
        # Encode the audio sequence
        audio_emb, a_mask = self.pretrained_models["audio_encoder"](audio, mask=a_mask)
                
        glob_emb, seq_emb, cond_mask = self.encoder.encode_audio(
            a_seq=audio_emb, a_mask=a_mask)
        
        latent = self.style_proj(glob_emb)            
        cond_emb = torch.cat((latent, seq_emb), dim=1)
        cond_length = cond_emb.size(1)
        
        pred_tokens = []
        for _ in range(num_tokens):
            output = self.decoder(input_embeds, cond=cond_emb, cond_mask=cond_mask, type=task, part=part)
            raw_pred_logit = output[:, -1:] # [1, 1, C]
            # For musix-to-motion task, we ignore <EOS> and <PAD> during generation, so we minimize the probability
            pred_logit = minimize_special_token_logits(logits=raw_pred_logit.clone(), k=3)
            if topk == 1:
                # Sample the token with highest probability
                pred_logit = F.softmax(pred_logit.clone(), dim=-1)
                pred_token = pred_logit.argmax(dim=-1)
            else:
                # Sample one token from tokens with top-k probability
                pred_token = self.sample_one_token(
                    raw_pred_logit=raw_pred_logit.clone(), 
                    topk=topk, sas=sas, 
                    temperature=temperature, 
                    task=task, part=part)
            
            pred_tokens.append(pred_token)
            pred_emb = self.quantizers["{:s}_{:s}".format(task, part)].get_codebook_entry(pred_token-3).contiguous().view(1, 1, -1) # [1, 1, C]
            input_embeds = torch.cat([input_embeds, pred_emb], dim=1)
        
        return torch.cat(pred_tokens, dim=1), input_embeds
    
    @torch.no_grad()
    def generate_motion_tokens_from_speech(
        self, audio, emotion, speaker_id, input_embeds, 
        num_tokens=1, task="s2m", part="body", 
        topk=1, sas=False, temperature=1.0, 
    ):
        # Encode the audio sequence
        speech_emb, speech_mask = self.pretrained_models["speech_encoder"](audio, mask=None)
        
        glob_emb, seq_emb, cond_mask = self.encoder.encode_speech(
            s_seq=speech_emb, emo=emotion, ids=speaker_id, s_mask=speech_mask)
        
        latent = self.style_proj(glob_emb)
        cond_emb = torch.cat((latent, seq_emb), dim=1)
        cond_length = cond_emb.size(1)
        
        pred_tokens = []
        for _ in range(num_tokens):
            output = self.decoder(input_embeds, cond=cond_emb, cond_mask=cond_mask, type=task, part=part)
            raw_pred_logit = output[:, -1:] # [1, 1, C]
            # For musix-to-motion task, we ignore <EOS> and <PAD> during generation, so we minimize the probability
            pred_logit = minimize_special_token_logits(logits=raw_pred_logit.clone(), k=3)
            if topk == 1:
                # Sample the token with highest probability
                pred_logit = F.softmax(pred_logit.clone(), dim=-1)
                pred_token = pred_logit.argmax(dim=-1)
            else:
                # Sample one token from tokens with top-k probability
                pred_token = self.sample_one_token(
                    raw_pred_logit=raw_pred_logit.clone(), 
                    topk=topk, sas=sas, 
                    temperature=temperature, 
                    task=task, part=part)
            
            pred_tokens.append(pred_token)
            pred_emb = self.quantizers["{:s}_{:s}".format(task, part)].get_codebook_entry(pred_token-3).contiguous().view(1, 1, -1) # [1, 1, C]
            input_embeds = torch.cat([input_embeds, pred_emb], dim=1)
        
        return torch.cat(pred_tokens, dim=1), input_embeds
    
    @torch.no_grad()
    def generate_next_motion_token_from_audio(
        self, audio, input_embeds, 
        task="a2m", part="body"
    ):
        # Encode the audio and motion primitives as cross-modality embedding
        a_mask = torch.ones(audio.size(0), audio.size(1)).bool().to(self.device)
        # Encode the audio sequence
        audio_emb, a_mask = self.pretrained_models["audio_encoder"](audio, mask=a_mask)
                
        glob_emb, seq_emb, cond_mask = self.encoder.encode_audio(
            a_seq=audio_emb, a_mask=a_mask)
        
        latent = self.style_proj(glob_emb)            
        cond_emb = torch.cat((latent, seq_emb), dim=1)
        cond_length = cond_emb.size(1)
        
        # Decode using GPT architecture
        output = self.decoder(input_embeds, cond=cond_emb, cond_mask=cond_mask, type=task, part=part)
        logit = output[:, -1:] # [1, 1, C]
        return logit
    
    @torch.no_grad()
    def generate_next_motion_token_from_speech(
        self, audio, emotion, speaker_id, input_embeds, 
        task="s2m", part="body"
    ):
        # Encode the audio sequence
        speech_emb, speech_mask = self.pretrained_models["speech_encoder"](audio, mask=None)
        
        glob_emb, seq_emb, cond_mask = self.encoder.encode_speech(
            s_seq=speech_emb, emo=emotion, ids=speaker_id, s_mask=speech_mask)
        
        latent = self.style_proj(glob_emb)
        cond_emb = torch.cat((latent, seq_emb), dim=1)
        cond_length = cond_emb.size(1)
        
        # Decode using GPT architecture
        output = self.decoder(input_embeds, cond=cond_emb, cond_mask=cond_mask, type=task, part=part)
        logit = output[:, -1:] # [1, 1, C]
        return logit
    
    @torch.no_grad()
    def sample_one_token(self, raw_pred_logit, topk=1, sas=False, temperature=1.0, task="t2m", part="body"):
        """
        :param raw_pred_logit: [batch_size, seq_len, num_dim]
        """
        if not sas:
            # Sample one token from tokens with top-k probability
            pred_logit = top_k_logits(raw_pred_logit.clone(), k=topk)
            pred_logit = F.softmax(pred_logit, dim=-1)
            pred_token = torch.multinomial(pred_logit[:, 0], num_samples=1)
        else:
            pred_logit = F.softmax(raw_pred_logit.clone(), dim=-1)
            _, idx_out = torch.topk(pred_logit, k=1, dim=-1)         # Deterministic
            if idx_out > self.pad_id:
                pred_token, *_ = self.quantizers["{:s}_{:s}".format(task, part)].multinomial(
                    idx_out.squeeze(dim=-1)-3, K=topk, temperature=temperature)
                pred_token += 3
            else:
                pred_token = idx_out.squeeze(dim=-1)
        return pred_token
    
if __name__ == "__main__":
    import yaml, json
    from networks.ude.seqvq import Quantizer
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with open("texts.json", "r") as f:
        texts = json.load(f)
    
    with open("configs/ude/config_ude_exp6.yaml", "r") as f:
        conf = yaml.safe_load(f)
    model = UDEModel(conf=conf["model"]["ude"]).to(device)
    quantizer = Quantizer(**conf["model"]["vqvae"]["t2m"]["body"]["quantizer"]).to(device)
    model.setup_quantizer(quantizer=quantizer, name="s2m_body")
    
    audio = torch.randn(2, 32000).float().to(device)
    tokens = torch.randint(0, 1024, size=(2, 10)).to(device) + 3
    # model.text_to_motion(text=texts[:2], input_ids=tokens)
    # model.generate_text_to_motion(text=texts[:1], max_num_tokens=50, topk=10, sas=True)
    # model.generate_audio_to_motion(audio=audio[:1], tokens=tokens[:1], max_num_tokens=100)
    model.generate_speech_to_motion(audio=audio[:1], emotion=None, speaker_id=None, tokens=tokens[:1], max_num_tokens=20)
    model.speech_to_motion(audio=audio, emotion=None, ids=None, input_ids_dict={"body": tokens})