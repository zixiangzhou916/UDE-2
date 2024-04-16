import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        self.register_buffer('positional_encoding', self.encoding)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        return self.encoding[:seq_len, :].clone().detach().to(x.device) + x
    
class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            # out = fused_leaky_relu(out, self.bias * self.lr_mul)  # DEBUG!!!

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )
        
class ScaledStylizationBlock(nn.Module):
    def __init__(self, emb_dim, latent_dim, dropout):
        super(ScaledStylizationBlock, self).__init__()
        self.latent_emb_layers = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(latent_dim, 2 * emb_dim)
        )
        self.emb_norm = nn.LayerNorm(emb_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(), 
            nn.Dropout(p=dropout), 
            nn.Linear(emb_dim, emb_dim)
        )
        
    def forward(self, emb, latent):
        """
        :param emb: [batch_size, nframes, dim]
        :param latent: [batch_size, dim]
        """
        lat_out = self.latent_emb_layers(latent)    # [bs, 1, dim] -> [bs, 1, 2*dim]
        scale, shift = torch.chunk(lat_out, 2, dim=2)
        h = self.emb_norm(emb) * (1 + scale) + shift
        h = self.out_layers(h)
        return h
    
class AttentiveStylizationBlock(nn.Module):
    def __init__(self, emb_dim, latent_dim, dropout):
        """
        :param emb_dim: dimension of query
        :param latent_dim: dimension of key
        """
        super(AttentiveStylizationBlock, self).__init__()
        self.q_layer = nn.Linear(emb_dim, emb_dim)
        self.k_layer = nn.Linear(latent_dim, emb_dim)
        self.v_layer = nn.Linear(latent_dim, emb_dim)
        self.softmax = nn.Softmax(dim=1)
        self.layer_norm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.dim = emb_dim
        
        self.q_layer.apply(init_weight)
        self.k_layer.apply(init_weight)
        self.v_layer.apply(init_weight)
    
    def forward(self, emb, latent):
        """
        :param emb: [batch_size, nframes, dim], query tensor
        :param latent: [batch_size, dim], key and value tensor
        """
        query_tensor = self.q_layer(emb)                    # [bs, seq_len, dim]
        val_tensor = self.v_layer(latent)                   # [bs, 1, dim]  
        key_tensor = self.k_layer(latent)                   # [bs, 1, dim]
        
        weights = torch.matmul(query_tensor, key_tensor.transpose(1, 2)) / np.sqrt(self.dim)    # [bs, seq_len, 1]
        weights = self.softmax(weights)  # [bs, seq_len, 1]
        
        pred = torch.matmul(weights, val_tensor)    # [bs, seq_len, dim]
        
        return self.layer_norm(pred + emb)
    
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, d_latent, dropout=0.1, style_module="scaled"):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        if style_module == "scaled":
            self.proj_out = ScaledStylizationBlock(emb_dim=d_model, latent_dim=d_latent, dropout=dropout)
        elif style_module == "attn":
            self.proj_out = AttentiveStylizationBlock(emb_dim=d_model, latent_dim=d_latent, dropout=dropout)

    def forward(self, q, k, v, latent=None, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))    # [batch_size, nframes, dim]
        
        if latent is not None:
            q = self.proj_out(emb=q, latent=latent)
            
        q += residual
        q = self.layer_norm(q)
        
        return q, attn
    
class FFN(nn.Module):
    
    def __init__(self, emb_dim, latent_dim, ffn_dim, dropout, style_module="scaled"):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(emb_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, emb_dim)
        self.layer_norm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        if style_module == "scaled":
            self.proj_out = ScaledStylizationBlock(emb_dim=emb_dim, latent_dim=latent_dim, dropout=dropout)
        elif style_module == "attn":
            self.proj_out = AttentiveStylizationBlock(emb_dim=emb_dim, latent_dim=latent_dim, dropout=dropout)

    def forward(self, x, emb=None):
        
        residual = x
        x = self.linear2(self.activation(self.linear1(x)))
        x = self.dropout(x)
        
        if emb is None:
            y = residual + x
        else:
            y = residual + self.proj_out(x, emb)
        
        y = self.layer_norm(y)
        
        # y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # y = x + self.proj_out(y, emb)
        return y

class AttLayer(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(AttLayer, self).__init__()
        self.W_q = nn.Linear(query_dim, value_dim)
        self.W_k = nn.Linear(key_dim, value_dim, bias=False)
        self.W_v = nn.Linear(key_dim, value_dim)

        self.softmax = nn.Softmax(dim=1)
        self.dim = value_dim

        self.W_q.apply(init_weight)
        self.W_k.apply(init_weight)
        self.W_v.apply(init_weight)

    def forward(self, query, key_mat):
        '''
        query (batch, query_dim)
        key (batch, seq_len, key_dim)
        '''
        # print(query.shape)
        query_vec = self.W_q(query).unsqueeze(-1)       # (batch, value_dim, 1)
        val_set = self.W_v(key_mat)                     # (batch, seq_len, value_dim)
        key_set = self.W_k(key_mat)                     # (batch, seq_len, value_dim)

        weights = torch.matmul(key_set, query_vec) / np.sqrt(self.dim)

        co_weights = self.softmax(weights)              # (batch, seq_len, 1)
        values = val_set * co_weights                   # (batch, seq_len, value_dim)
        pred = values.sum(dim=1)                        # (batch, value_dim)
        return pred, co_weights
 
class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, d_latent, n_head, d_k, d_v, dropout=0.1, style_module="naive"):
        super(DecoderLayer, self).__init__()
        assert style_module in ["scaled", "attn"]
        
        self.slf_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, 
                                           d_k=d_k, d_v=d_v, 
                                           d_latent=d_latent, 
                                           dropout=dropout, 
                                           style_module=style_module)
        self.crs_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, 
                                           d_k=d_k, d_v=d_v, 
                                           d_latent=d_latent, 
                                           dropout=dropout, 
                                           style_module=style_module)
        self.ffn = FFN(emb_dim=d_model, latent_dim=d_latent, ffn_dim=d_inner, 
                       dropout=dropout, style_module=style_module)
        
    def forward(self, dec_input, enc_output, latent=None, slf_attn_mask=None, crs_attn_mask=None):
        
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, latent, mask=slf_attn_mask)
        dec_output, dec_crs_attn = self.crs_attn(dec_output, enc_output, enc_output, latent, mask=crs_attn_mask)
        dec_output = self.ffn(dec_output, latent)
        
        return dec_output, dec_slf_attn, dec_crs_attn

"""Condition encoder, it encodes text, audio, and speech, respectively. 
"""
class EncoderModel(nn.Module):
    def __init__(
        self, 
        d_audio: int, 
        d_text: int, 
        d_speech: int, 
        d_word: int, 
        n_emo: int, 
        n_ids: int, 
        d_model: int, 
        d_inner: int, 
        n_head: int, 
        n_layers: int, 
        dropout: float = 0.1, 
        max_seq_length: int = 128, 
        **kwargs
    ):
        """
        :param n_tokens: number of motion tokens
        :param d_audio: input dimention of audio features
        :param n_layers: number of cross modal transformer layers
        :param n_head: number of heads of self-attention
        :param d_model: dimension of input to transformer encoder
        :param d_inner: dimension of intermediate layer of transformer encoder
        :param dropout: dropout rate
        :param **kwrags: any other possible arguments (optional)
        """
        super(EncoderModel, self).__init__()
        
        self.max_seq_length = max_seq_length
        self.position_enc = PositionalEncoding(d_model, max_len=5000)
        self.pad_embedding = nn.Parameter(torch.randn(1, d_model), requires_grad=True)
        self.emo_emb = nn.Embedding(n_emo, d_model)         # emotion embedding
        self.id_emb = nn.Embedding(n_ids, d_model)          # ID embedding
        
        # self.a_proj = nn.Sequential(
        #     nn.Conv1d(d_audio, 512, kernel_size=4, stride=2, padding=1), 
        #     nn.LeakyReLU(0.2, inplace=True), 
        #     nn.Conv1d(512, d_model, kernel_size=4, stride=2, padding=1)
        # )
        self.a_proj = nn.Linear(d_audio, d_model, bias=False)   # music embedding projection
        self.t_proj = nn.Linear(d_text, d_model, bias=False)    # text embedding projection
        self.m_proj = nn.Linear(d_model, d_model, bias=False)   # motion embedding projection
        self.s_proj = nn.Sequential(
            nn.Conv1d(d_speech, 512, kernel_size=4, stride=2, padding=1), 
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Conv1d(512, d_model, kernel_size=4, stride=2, padding=1)
        )
        # self.s_proj = nn.Linear(d_speech, d_model, bias=False)  # speech embedding projection
        self.w_proj = nn.Linear(d_word, d_model, bias=False)    # word embedding projection
        
        self.a_token = nn.Parameter(torch.randn(d_model), requires_grad=True)   # music token
        self.m_token = nn.Parameter(torch.randn(d_model), requires_grad=True)   # motion token
        self.t_token = nn.Parameter(torch.randn(d_model), requires_grad=True)   # text token
        self.s_token = nn.Parameter(torch.randn(d_model), requires_grad=True)   # speech token
        self.w_token = nn.Parameter(torch.randn(d_model), requires_grad=True)   # word token
        self.id_token = nn.Parameter(torch.randn(d_model), requires_grad=True)  # ID token
        self.emo_token = nn.Parameter(torch.randn(d_model), requires_grad=True) # emotion token
        
        self.agg_t_token = nn.Parameter(torch.randn(d_model), requires_grad=True)   # t2m task token
        self.agg_a_token = nn.Parameter(torch.randn(d_model), requires_grad=True)   # a2m task token
        self.agg_s_token = nn.Parameter(torch.randn(d_model), requires_grad=True)   # s2m task token

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=d_inner, 
            dropout=dropout, 
            activation="gelu")
        self.transformer = nn.TransformerEncoder(
            encoder_layer=transformer_layer, 
            num_layers=n_layers)
        
    def encode_text(self, t_seq, t_mask):
        """
        :param t_seq: [batch_size, nframes, dim], the text sequence
        :param t_mask: [batch_size, nframes], the mask of text sequence
        """
        t_seq = self.t_proj(t_seq)

        agg_token = self.agg_t_token[None, None].repeat(t_seq.shape[0], 1, 1)
        t_token = self.t_token[None, None].repeat(t_seq.shape[0], t_seq.shape[1], 1)
        token_mask = torch.ones(t_seq.shape[0], 1).bool().to(t_seq.device)

        pad_length = self.max_seq_length - t_seq.size(1) - 1
        if pad_length > 0:
            pad_embedding = self.pad_embedding[:, None].repeat(t_seq.size(0), pad_length, 1)
            pad_mask = torch.zeros(t_seq.size(0), pad_length).bool().to(t_seq.device)
            
            input_embeds = torch.cat([agg_token, t_seq+t_token, pad_embedding], dim=1)
            attn_mask = torch.cat([token_mask, t_mask, pad_mask], dim=1)
        else:
            input_embeds = torch.cat((agg_token, t_seq + t_token), dim=1)
            attn_mask = torch.cat([token_mask, t_mask], dim=1)
                
        input_embeds = self.position_enc(input_embeds)
        input_embeds = input_embeds.permute(1, 0, 2)    # [nframes+1, bs, dim]
        
        hidden_states = self.transformer(input_embeds, src_key_padding_mask=~attn_mask)
        hidden_states = hidden_states.permute(1, 0, 2)    # [bs, nframes+1, dim]
        glob_emb, seq_emb = hidden_states[:, :1], hidden_states[:, 1:]
        
        return glob_emb, seq_emb, attn_mask
    
    def encode_audio(self, a_seq, a_mask):
        """
        :param a_seq: [batch_size, nframes_audio, dim], the audio sequence
        :param a_mask: [batch_size, nframes_audio], the mask of audio sequence
        """
        # a_seq = self.a_proj(a_seq.permute(0, 2, 1)).permute(0, 2, 1)        
        a_seq = self.a_proj(a_seq)        
        a_token = self.a_token[None, None].repeat(a_seq.shape[0], a_seq.shape[1], 1)
            
        agg_token = self.agg_a_token[None, None].repeat(a_seq.shape[0], 1, 1)
        agg_mask = torch.ones(a_seq.shape[0], 1).bool().to(a_seq.device)
        
        pad_length = self.max_seq_length - a_seq.size(1) - 1
        if pad_length > 0:
            pad_embedding = self.pad_embedding[:, None].repeat(a_seq.size(0), pad_length, 1)
            pad_mask = torch.zeros(a_seq.size(0), pad_length).bool().to(a_seq.device)
            
            input_embeds = torch.cat([agg_token, a_seq + a_token, pad_embedding], dim=1)
            attn_mask = torch.cat([agg_mask, a_mask, pad_mask], dim=1)
        else:
            input_embeds = torch.cat([agg_token, a_seq + a_token], dim=1)
            attn_mask = torch.cat([agg_mask, a_mask], dim=1)
                
        input_embeds = self.position_enc(input_embeds)
        input_embeds = input_embeds.permute(1, 0, 2)    # [nframes_audio+nframes_motion+1, bs, dim]

        hidden_states = self.transformer(input_embeds, src_key_padding_mask=~attn_mask)
        hidden_states = hidden_states.permute(1, 0, 2)    # [bs, nframes_audio+nframes_motion+1, dim]
        glob_emb, seq_emb = hidden_states[:, :1], hidden_states[:, 1:]

        return glob_emb, seq_emb, attn_mask
    
    def encode_speech(self, s_seq, emo, ids, s_mask):
        # Conv1d to align HuBert features
        s_seq = self.s_proj(s_seq.permute(0, 2, 1)).permute(0, 2, 1)
        s_token = self.s_token[None, None].repeat(s_seq.shape[0], s_seq.shape[1], 1)
        seq_len = s_seq.size(1)
        s_mask = torch.ones(s_seq.size(0), seq_len).bool().to(s_seq.device)
        
        if emo is not None:
            if emo.dim() == 1:
                emo = emo.unsqueeze(dim=-1)
            emo_seq = self.emo_emb(emo)
            emo_token = self.emo_token[None, None].repeat(emo.size(0), 1, 1)
            emo_mask = torch.ones(emo_seq.size(0), 1).bool().to(s_seq.device)
            seq_len += 1
        if ids is not None:
            if ids.dim() == 1:
                ids = ids.unsqueeze(dim=-1)
            ids_seq = self.id_emb(ids)
            ids_token = self.id_token[None, None].repeat(ids.size(0), 1, 1)
            ids_mask = torch.ones(ids_seq.size(0), 1).bool().to(s_seq.device)
            seq_len += 1
            
        agg_token = self.agg_s_token[None, None].repeat(s_seq.shape[0], 1, 1)
        agg_mask = torch.ones(s_seq.shape[0], 1).bool().to(s_seq.device)
        
        pad_length = self.max_seq_length - seq_len - 1
        
        input_embeds = torch.cat([agg_token, s_seq + s_token], dim=1)
        attn_mask = torch.cat([agg_mask, s_mask], dim=1)
        if emo is not None:
            input_embeds = torch.cat([input_embeds, emo_seq+emo_token], dim=1)
            attn_mask = torch.cat([attn_mask, emo_mask], dim=1)
        if ids is not None:
            input_embeds = torch.cat([input_embeds, ids_seq+ids_token], dim=1)
            attn_mask = torch.cat([attn_mask, ids_mask], dim=1)
        if pad_length > 0:
            pad_embedding = self.pad_embedding[:, None].repeat(s_seq.size(0), pad_length, 1)
            pad_mask = torch.zeros(s_seq.size(0), pad_length).bool().to(s_seq.device)
            
            input_embeds = torch.cat([input_embeds, pad_embedding], dim=1)
            attn_mask = torch.cat([attn_mask, pad_mask], dim=1)
        
        input_embeds = self.position_enc(input_embeds)
        input_embeds = input_embeds.permute(1, 0, 2)    # [nframes_audio+nframes_motion+1, bs, dim]

        hidden_states = self.transformer(input_embeds, src_key_padding_mask=~attn_mask)
        hidden_states = hidden_states.permute(1, 0, 2)    # [bs, nframes_audio+nframes_motion+1, dim]
        glob_emb, seq_emb = hidden_states[:, :1], hidden_states[:, 1:]

        return glob_emb, seq_emb, attn_mask

class EncoderModelV2(nn.Module):
    def __init__(
        self, 
        d_audio: int, 
        d_text: int, 
        d_speech: int, 
        d_word: int, 
        n_emo: int, 
        n_ids: int, 
        d_model: int, 
        d_inner: int, 
        n_head: int, 
        n_layers: int, 
        dropout: float = 0.1, 
        max_seq_length: int = 128, 
        **kwargs
    ):
        """
        :param n_tokens: number of motion tokens
        :param d_audio: input dimention of audio features
        :param n_layers: number of cross modal transformer layers
        :param n_head: number of heads of self-attention
        :param d_model: dimension of input to transformer encoder
        :param d_inner: dimension of intermediate layer of transformer encoder
        :param dropout: dropout rate
        :param **kwrags: any other possible arguments (optional)
        """
        super(EncoderModelV2, self).__init__()
        
        self.max_seq_length = max_seq_length
        self.position_enc = PositionalEncoding(d_model, max_len=5000)
        self.pad_embedding = nn.Parameter(torch.randn(1, d_model), requires_grad=True)
        self.emo_emb = nn.Embedding(n_emo, d_model)         # emotion embedding
        self.id_emb = nn.Embedding(n_ids, d_model)          # ID embedding
        
        self.a_proj = nn.Linear(d_audio, d_model, bias=False)   # music embedding projection
        self.t_proj = nn.Linear(d_text, d_model, bias=False)    # text embedding projection
        self.m_proj = nn.Linear(d_model, d_model, bias=False)   # motion embedding projection
        self.s_proj = nn.Linear(d_speech, d_model, bias=False)  # speech embedding projection
        self.w_proj = nn.Linear(d_word, d_model, bias=False)    # word embedding projection
        
        self.a_token = nn.Parameter(torch.randn(d_model), requires_grad=True)   # music token
        self.m_token = nn.Parameter(torch.randn(d_model), requires_grad=True)   # motion token
        self.t_token = nn.Parameter(torch.randn(d_model), requires_grad=True)   # text token
        self.s_token = nn.Parameter(torch.randn(d_model), requires_grad=True)   # speech token
        self.w_token = nn.Parameter(torch.randn(d_model), requires_grad=True)   # word token
        self.id_token = nn.Parameter(torch.randn(d_model), requires_grad=True)  # ID token
        self.emo_token = nn.Parameter(torch.randn(d_model), requires_grad=True) # emotion token
        
        self.agg_t_token = nn.Parameter(torch.randn(d_model), requires_grad=True)   # t2m task token
        self.agg_a_token = nn.Parameter(torch.randn(d_model), requires_grad=True)   # a2m task token
        self.agg_s_token = nn.Parameter(torch.randn(d_model), requires_grad=True)   # s2m task token

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=d_inner, 
            dropout=dropout, 
            activation="gelu")
        self.transformer = nn.TransformerEncoder(
            encoder_layer=transformer_layer, 
            num_layers=n_layers)
    
    def encode_text(self, t_seq, t_mask):
        """
        :param t_seq: [batch_size, nframes, dim], the text sequence
        :param t_mask: [batch_size, nframes], the mask of text sequence
        """
        t_seq = self.t_proj(t_seq)

        agg_token = self.agg_t_token[None, None].repeat(t_seq.shape[0], 1, 1)
        t_token = self.t_token[None, None].repeat(t_seq.shape[0], t_seq.shape[1], 1)
        token_mask = torch.ones(t_seq.shape[0], 1).bool().to(t_seq.device)

        pad_length = self.max_seq_length - t_seq.size(1) - 1
        if pad_length > 0:
            pad_embedding = self.pad_embedding[:, None].repeat(t_seq.size(0), pad_length, 1)
            pad_mask = torch.zeros(t_seq.size(0), pad_length).bool().to(t_seq.device)
            
            input_embeds = torch.cat([agg_token, t_seq+t_token, pad_embedding], dim=1)
            attn_mask = torch.cat([token_mask, t_mask, pad_mask], dim=1)
        else:
            input_embeds = torch.cat((agg_token, t_seq + t_token), dim=1)
            attn_mask = torch.cat([token_mask, t_mask], dim=1)
                
        input_embeds = self.position_enc(input_embeds)
        input_embeds = input_embeds.permute(1, 0, 2)    # [nframes+1, bs, dim]
        
        hidden_states = self.transformer(input_embeds, src_key_padding_mask=~attn_mask)
        hidden_states = hidden_states.permute(1, 0, 2)    # [bs, nframes+1, dim]
        glob_emb, seq_emb = hidden_states[:, :1], hidden_states[:, 1:]
        
        return glob_emb, seq_emb, attn_mask
    
    def encode_audio(self, a_seq, a_mask):
        """
        :param a_seq: [batch_size, nframes_audio, dim], the audio sequence
        :param a_mask: [batch_size, nframes_audio], the mask of audio sequence
        """
        # a_seq = self.a_proj(a_seq.permute(0, 2, 1)).permute(0, 2, 1)        
        a_seq = self.a_proj(a_seq)        
        a_token = self.a_token[None, None].repeat(a_seq.shape[0], a_seq.shape[1], 1)
            
        agg_token = self.agg_a_token[None, None].repeat(a_seq.shape[0], 1, 1)
        agg_mask = torch.ones(a_seq.shape[0], 1).bool().to(a_seq.device)
        
        pad_length = self.max_seq_length - a_seq.size(1) - 1
        if pad_length > 0:
            pad_embedding = self.pad_embedding[:, None].repeat(a_seq.size(0), pad_length, 1)
            pad_mask = torch.zeros(a_seq.size(0), pad_length).bool().to(a_seq.device)
            
            input_embeds = torch.cat([agg_token, a_seq + a_token, pad_embedding], dim=1)
            attn_mask = torch.cat([agg_mask, a_mask, pad_mask], dim=1)
        else:
            input_embeds = torch.cat([agg_token, a_seq + a_token], dim=1)
            attn_mask = torch.cat([agg_mask, a_mask], dim=1)
                
        input_embeds = self.position_enc(input_embeds)
        input_embeds = input_embeds.permute(1, 0, 2)    # [nframes_audio+nframes_motion+1, bs, dim]

        hidden_states = self.transformer(input_embeds, src_key_padding_mask=~attn_mask)
        hidden_states = hidden_states.permute(1, 0, 2)    # [bs, nframes_audio+nframes_motion+1, dim]
        glob_emb, seq_emb = hidden_states[:, :1], hidden_states[:, 1:]

        return glob_emb, seq_emb, attn_mask
    
    def encode_speech(self, s_seq, emo, ids, s_mask):
        # 
        s_seq = self.s_proj(s_seq)
        s_token = self.s_token[None, None].repeat(s_seq.shape[0], s_seq.shape[1], 1)
        seq_len = s_seq.size(1)
        s_mask = torch.ones(s_seq.size(0), seq_len).bool().to(s_seq.device)
        
        if emo is not None:
            if emo.dim() == 1:
                emo = emo.unsqueeze(dim=-1)
            emo_seq = self.emo_emb(emo)
            emo_token = self.emo_token[None, None].repeat(emo.size(0), 1, 1)
            emo_mask = torch.ones(emo_seq.size(0), 1).bool().to(s_seq.device)
            seq_len += 1
        if ids is not None:
            if ids.dim() == 1:
                ids = ids.unsqueeze(dim=-1)
            ids_seq = self.id_emb(ids)
            ids_token = self.id_token[None, None].repeat(ids.size(0), 1, 1)
            ids_mask = torch.ones(ids_seq.size(0), 1).bool().to(s_seq.device)
            seq_len += 1
            
        agg_token = self.agg_s_token[None, None].repeat(s_seq.shape[0], 1, 1)
        agg_mask = torch.ones(s_seq.shape[0], 1).bool().to(s_seq.device)
        
        pad_length = self.max_seq_length - seq_len - 1
        
        input_embeds = torch.cat([agg_token, s_seq + s_token], dim=1)
        attn_mask = torch.cat([agg_mask, s_mask], dim=1)
        if emo is not None:
            input_embeds = torch.cat([input_embeds, emo_seq+emo_token], dim=1)
            attn_mask = torch.cat([attn_mask, emo_mask], dim=1)
        if ids is not None:
            input_embeds = torch.cat([input_embeds, ids_seq+ids_token], dim=1)
            attn_mask = torch.cat([attn_mask, ids_mask], dim=1)
        if pad_length > 0:
            pad_embedding = self.pad_embedding[:, None].repeat(s_seq.size(0), pad_length, 1)
            pad_mask = torch.zeros(s_seq.size(0), pad_length).bool().to(s_seq.device)
            
            input_embeds = torch.cat([input_embeds, pad_embedding], dim=1)
            attn_mask = torch.cat([attn_mask, pad_mask], dim=1)
        
        input_embeds = self.position_enc(input_embeds)
        input_embeds = input_embeds.permute(1, 0, 2)    # [nframes_audio+nframes_motion+1, bs, dim]

        hidden_states = self.transformer(input_embeds, src_key_padding_mask=~attn_mask)
        hidden_states = hidden_states.permute(1, 0, 2)    # [bs, nframes_audio+nframes_motion+1, dim]
        glob_emb, seq_emb = hidden_states[:, :1], hidden_states[:, 1:]

        return glob_emb, seq_emb, attn_mask
