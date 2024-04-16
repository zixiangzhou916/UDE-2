"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""
import os
import sys
sys.path.append(os.getcwd())

import math
import logging
from typing import Dict
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import importlib

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        # self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        # self.lr_mul = lr_mul
        self.scale = (1 / math.sqrt(in_dim))
        self.lr_mul = 1.0

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )

""" Causal Attention & Padding Agnostice """    
class Block(nn.Module):
    def __init__(self, n_emb, n_head, block_size, attn_pdrop, resid_pdrop):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)
        self.attn = CausalCrossConditionalSelfAttention(
            n_emb=n_emb, n_head=n_head, block_size=block_size, 
            attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.GELU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(resid_pdrop),
        )
        
    def forward(self, x, cond_len, cond_mask, token_len=None):
        x = x + self.attn(self.ln1(x), cond_len=cond_len, cond_mask=cond_mask, token_len=token_len)
        x = x + self.mlp(self.ln2(x))
        return x
    
""" Causal Attention & Padding Aware """
class BlockV2(nn.Module):
    def __init__(self, n_emb, n_head, block_size, attn_pdrop, resid_pdrop):
        super(BlockV2, self).__init__()
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)
        self.attn = CausalCrossConditionalSelfAttentionV2(
            n_emb=n_emb, n_head=n_head, block_size=block_size, 
            attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.GELU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(resid_pdrop),
        )
        
    def forward(self, x, cond_len, token_len=None, padding_mask=None):
        x = x + self.attn(self.ln1(x), cond_len=cond_len, token_len=token_len, padding_mask=padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x
     
""" Full Attention & Padding Aware """
class BlockV3(nn.Module):
    def __init__(self, n_emb, n_head, block_size, attn_pdrop, resid_pdrop):
        super(BlockV3, self).__init__()
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)
        self.attn = FullCrossConditionalSelfAttention(
            n_emb=n_emb, n_head=n_head, block_size=block_size, 
            attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.GELU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(resid_pdrop),
        )
        
    def forward(self, x, cond_len, token_len=None, padding_mask=None):
        x = x + self.attn(self.ln1(x), cond_len=cond_len, token_len=token_len, padding_mask=padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x
     
""" Causal Attention & Padding Agnostice """          
class CausalCrossConditionalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """
    def __init__(self, n_emb, n_head, block_size, attn_pdrop, resid_pdrop):
        super(CausalCrossConditionalSelfAttention, self).__init__()
        assert n_emb % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_emb, n_emb)
        self.query = nn.Linear(n_emb, n_emb)
        self.value = nn.Linear(n_emb, n_emb)
        
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        
        # output projection
        self.proj = nn.Linear(n_emb, n_emb)
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        # self.mask = se
        self.n_head = n_head
        
    def forward(self, x, cond_len, cond_mask, token_len=None):
        """
        :param x: [batch_size, nframes, dim]
        :param cond_len: 
        :param cond_mask: [batch_size, nframes]
        """
        B, T, C = x.size()  # T = 3*t (music up down)
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        t = T - cond_len
        if token_len is None: 
            token_len = t
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                
        mask = torch.zeros(B, 1, T, T).float().to(x.device)
        # mask[:, :, :, :cond_len] = 1
        for i in range(B):
            mask[i, 0, :, :cond_mask[i].sum()] = 1
        
        for k in range(t // token_len):
            t_start = cond_len + k * token_len
            t_end = t_start + token_len
            mask[:, :, t_start:t_end, cond_len:] = \
                self.mask[:, :, :token_len, :token_len].repeat(
                    1, 1, 1, t // token_len)
        # mask[:, :, -t:, -t:] = self.mask[:, :, :t, :t]
        att = att.masked_fill(mask == 0, float('-inf'))
        
        # """ DEBUG """
        # import matplotlib
        # from matplotlib import colors
        # from matplotlib.ticker import PercentFormatter
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        
        # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10*2, 10*2), dpi=200)
        # axs[0,0].matshow(mask[0,0].data.cpu().numpy(), cmap=plt.cm.Spectral_r, interpolation='none', vmin=-1, vmax=1)
        # axs[0,1].matshow(att[0,0].data.cpu().numpy(), cmap=plt.cm.Spectral_r, interpolation='none', vmin=-1, vmax=1)
        # axs[1,0].matshow(mask[1,0].data.cpu().numpy(), cmap=plt.cm.Spectral_r, interpolation='none', vmin=-1, vmax=1)
        # axs[1,1].matshow(att[1,0].data.cpu().numpy(), cmap=plt.cm.Spectral_r, interpolation='none', vmin=-1, vmax=1)
        # plt.savefig("similarity.png")
        # plt.close()  
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

""" Causal Attention & Padding Aware """
class CausalCrossConditionalSelfAttentionV2(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """
    def __init__(self, n_emb, n_head, block_size, attn_pdrop, resid_pdrop):
        super(CausalCrossConditionalSelfAttentionV2, self).__init__()
        assert n_emb % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_emb, n_emb)
        self.query = nn.Linear(n_emb, n_emb)
        self.value = nn.Linear(n_emb, n_emb)
        
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        
        # outut projection
        self.proj = nn.Linear(n_emb, n_emb)
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        # self.mask = se
        self.n_head = n_head
        
    def forward(self, x, cond_len, token_len=None, padding_mask=None):
        """
        :param x: [batch_size, nframes, dim]
        :param cond_len: integer, length of condition sequence
        :param token_len: (optional) integer, length of token sequence
        :param padding_mask: (optional) [batch_size, nframes]
        """
        B, T, C = x.size()
        
        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)       # [B, nh, T, hs]
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # [B, nh, T, hs]
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # [B, nh, T, hs]
        t = T - cond_len    # length of token sequence
        if token_len is None:
            token_len = t
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # attention matrix
        
        # We generate causal attention mask as: https://github.com/huggingface/transformers/issues/9366
        # and padding mask as: https://wikidocs.net/167210
        # Get the causal attention mask
        causal_mask = torch.zeros(B, 1, T, T).float().to(x.device)
        causal_mask[:, :, :, :cond_len] = 1
        for k in range(t // token_len):
            t_start = cond_len + k * token_len
            t_end = t_start + token_len
            causal_mask[:, :, t_start:t_end, cond_len:] = \
                self.mask[:, :, :token_len, :token_len].repeat(
                    1, 1, 1, t // token_len)
        
        # Get the padding mask
        if padding_mask is not None:
            padded_mask = torch.zeros(B, 1, T, T).float().to(x.device)
            padded_mask[:, :, cond_len:, cond_len:] = 1.    # Correspond to trg_seq
            for b, pmask in enumerate(padding_mask):    # pmask: [seq_len]
                # n = int(pmask.size(0) - pmask.sum().item()) # This is wrong!!!
                n = int(pmask.sum().item())
                padded_mask[b, 0, :, :n] = 1.
            
            final_mask = causal_mask * padded_mask
        else:
            final_mask = causal_mask.clone()
        
        # Apply mask
        att = att.masked_fill(final_mask == 0, float('-inf'))        
        att = F.softmax(att, dim=-1)    # To avoid NaN, we substract the att by the maximum value            
        att = self.attn_drop(att)
        y = att @ v     # [B, nh, T, T] x [B, nh, T, hs] -> [B, nh, T, hs]
        y = y.transpose(1, 2).contiguous().view(B, T, C)    # re-assemble all head outputs side by side
        
        # output projection
        y = self.resid_drop(self.proj(y))
                                
        return y

""" Full Attention & Padding Aware """
class FullCrossConditionalSelfAttention(nn.Module):
    def __init__(self, n_emb, n_head, block_size, attn_pdrop, resid_pdrop):
        super(FullCrossConditionalSelfAttention, self).__init__()
        assert n_emb % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_emb, n_emb)
        self.query = nn.Linear(n_emb, n_emb)
        self.value = nn.Linear(n_emb, n_emb)
        
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        
        # output projection
        self.proj = nn.Linear(n_emb, n_emb)
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        # self.mask = se
        self.n_head = n_head
        
    def forward(self, x, cond_len, token_len=None, padding_mask=None):
        """
        :param x: [batch_size, nframes, dim]
        :param cond_len: integer, length of condition sequence
        :param token_len: (optional) integer, length of token sequence
        :param padding_mask: (optional) [batch_size, nframes]
        """
        B, T, C = x.size()
        
        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        t = T - cond_len
        if token_len is None: 
            token_len = t
            
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # attention matrix
        
        # We generate padding mask as: https://wikidocs.net/167210
        if padding_mask is not None:
            padded_mask = torch.zeros(B, 1, T, T).float().to(x.device)
            padded_mask[:, :, :, cond_len:] = 1.
            for b, pmask in enumerate(padding_mask):
                n = int(pmask.size(0) - pmask.sum().item())
                padded_mask[b, 0, :, :n] = 1.
            
        else:
            padded_mask = torch.ones(B, 1, T, T).float().to(x.device)
        
        # Apply mask
        att = att.masked_fill(padded_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v     # [B, nh, T, T] x [B, nh, T, hs] -> [B, nh, T, hs]
        y = y.transpose(1, 2).contiguous().view(B, T, C)    # re-assemble all head outputs side by side
        
        # output projection
        y = self.resid_drop(self.proj(y))
                
        return y

""" Causal Attention & Padding Agnostic """
class CrossCondGPTHead(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    
    def __init__(self, conf):
        super(CrossCondGPTHead, self).__init__()
        self.conf = conf
        # transformer
        self.blocks = nn.ModuleList()
        for _ in range(conf["n_layers"]):
            self.blocks.append(
                Block(n_emb=conf["d_latent"], n_head=conf["n_head"], block_size=conf["block_size"], 
                      attn_pdrop=conf["attn_pdrop"], resid_pdrop=conf["resid_pdrop"])
            )

        # decoder head
        self.ln_f = nn.LayerNorm(conf["d_latent"])
        self.block_size = conf["block_size"]
        self.head = nn.Linear(conf["d_latent"], conf["n_tokens"], bias=False) # 2048
        
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, x, cond_mask, target_length, padding_mask=None):
        
        _, T, _ = x.shape
        for block in self.blocks:
            x = block(x, cond_len=T-target_length, cond_mask=cond_mask, token_len=target_length)
        x = self.ln_f(x)
        N, T, C = x.size()
        logits = self.head(x[:, -target_length:])
        
        return logits

""" 
1. Causal Attention 
2. Padding Agnostic 
3. Domain Agnostic"""
class CrossCondGPTBase(nn.Module):
    """ the full GPT language model, with a context size of block_size """
    
    def __init__(self, conf):
        super(CrossCondGPTBase, self).__init__()
        
        self.ids_emb = nn.Linear(conf["d_ids_model"], conf["d_latent"])
        self.pos_emb = nn.Parameter(torch.zeros(1, conf["n_positions"], conf["d_latent"]), requires_grad=True)
        self.cond_emb = nn.Linear(conf["d_cond_model"], conf["d_latent"])
        self.drop = nn.Dropout(conf["drop"])
        # transformer
        self.blocks = nn.ModuleList()
        for _ in range(conf["n_layers"]):
            self.blocks.append(
                Block(n_emb=conf["d_latent"], n_head=conf["n_head"], block_size=conf["block_size"], 
                      attn_pdrop=conf["attn_pdrop"], resid_pdrop=conf["resid_pdrop"])
            )
        
        self.block_size = conf["block_size"]
        self.apply(self._init_weights)
        
    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # module.weight.data.uniform_(math.sqrt(6.0/sum(module.weight.size())))
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, idx, cond, cond_mask):
        
        b, t, _ = idx.shape
        _, cond_t, _ = cond.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        # if self.requires_head:
        input_embeddings = self.ids_emb(idx) # each index maps to a (learnable) vector
        input_embeddings = torch.cat([self.cond_emb(cond), input_embeddings], dim=1)

        pos_emb_1 = self.pos_emb[:, :cond_t, :]
        pos_emb_2 = self.pos_emb[:, cond_t:cond_t+t, :]
        position_embeddings = torch.cat([pos_emb_1, pos_emb_2], dim=1) # each position maps to a (learnable) vector
        
        x = self.drop(input_embeddings + position_embeddings)

        for block in self.blocks:
            x = block(x, cond_len=cond.shape[1], cond_mask=cond_mask)
        # x = self.ln_f(x)

        return x

""" 
1. Target tokens are domain-agnostic
2. Output tokens are domain-specific
"""
class CrossCondGPT(nn.Module):
    def __init__(self, conf):
        super(CrossCondGPT, self).__init__()
        self.base = importlib.import_module(
            conf["gpt_base"]["arch_path"], package="networks").__getattribute__(
                conf["gpt_base"]["arch_name"])(conf["gpt_base"])
        self.heads = nn.ModuleDict()
        for cat, cat_conf in conf["gpt_head"].items():
            for part, part_conf in cat_conf.items():
                self.heads["{:s}_{:s}".format(cat, part)] = \
                    importlib.import_module(
                        part_conf["arch_path"], package="networks").__getattribute__(
                            part_conf["arch_name"])(part_conf)   
    
    def forward(
        self, 
        idx: torch.Tensor, 
        cond: torch.Tensor, 
        cond_mask: torch.Tensor, 
        padding_mask: torch.Tensor = None, 
        type: str = "t2m", 
        part: str = "body"
    ):
        base_out = self.base(idx, cond, cond_mask)
        head_out = self.heads["{:s}_{:s}".format(type, part)](
            base_out, cond_mask=cond_mask, target_length=idx.size(1)) 
        return head_out
    
  
if __name__ == "__main__":
    import yaml
    with open("configs/ude/config_ude_exp1.yaml", "r") as f:
        conf = yaml.safe_load(f)
    
    conf = conf["model"]["ude"]["gpt"]
    
    gpt = importlib.import_module(
        conf["arch_path"], package="networks").__getattribute__(
            conf["arch_name"])(conf)
    # gpt.eval()
    
    cond = torch.randn(2, 32, 512)
    cond_mask = torch.ones(2, 32).bool()
    cond_mask[0, 20:] = False
    cond_mask[1, 25:] = False
    # tokens = torch.randint(2, 2048, (2, 32))
    input_embeds = torch.randn(2, 32, 512)
    padding_mask = torch.zeros(2, 32).float()
    padding_mask[0, 20:] = 1.0
    padding_mask[1, 10:] = 1.0
    output = gpt(idx=input_embeds, cond=cond, cond_mask=cond_mask, type="t2m", padding_mask=padding_mask)
    print('--- done ---')