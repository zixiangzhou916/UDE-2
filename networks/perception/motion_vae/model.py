import torch
import torch.nn as nn
import torch.nn.functional as F
# from .layers import *

def lengths_to_mask(lengths, device):
    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

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

class MotionEncoderV1(nn.Module):
    def __init__(self, d_input, d_model, d_inner, n_head, n_layer, dropout=0.1, activation="gelu", **kwargs):
        super(MotionEncoderV1, self).__init__()
        self.position_enc = PositionalEncoding(d_model, max_len=1000)
        self.linear = nn.Linear(d_input, d_model)
        
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                               nhead=n_head,
                                                               dim_feedforward=d_inner,
                                                               dropout=dropout,
                                                               activation=activation)
        self.encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, 
                                             num_layers=n_layer)
        
        self.mu = nn.Parameter(torch.randn(d_model), requires_grad=True)
        self.sigma = nn.Parameter(torch.randn(d_model), requires_grad=True)
        
    def forward(self, x, mask=None):
        """
        :param x: [batch_size, nframes, dim]
        """
        batch_size = x.shape[0]
        x_emb = self.linear(x)
        mu_token = self.mu[None, None].repeat(batch_size, 1, 1)
        sigma_token = self.sigma[None, None].repeat(batch_size, 1, 1)
        x_seq = torch.cat((mu_token, sigma_token, x_emb), dim=1)
        x_seq = self.position_enc(x_seq)
        x_seq = x_seq.permute(1, 0, 2)  # [nframes, batch_size, dim]
        
        if mask is not None:
            aux_mask = torch.ones((batch_size, 2), dtype=bool).to(x.device)
            mask = torch.cat((aux_mask, mask), dim=1)
            x_enc = self.encoder(x_seq, src_key_padding_mask=~mask)
        else:
            x_enc = self.encoder(x_seq)
            
        x_enc = x_enc.permute(1, 0, 2)  # [batch_size, nframes, dim]
        
        mu = x_enc[:, :1]
        logvar = x_enc[:, 1:2]
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        
        return dist
    
    def sample_from_distribution(self, distribution):
        return distribution.rsample()

class MotionDecoderV1(nn.Module):
    def __init__(self, d_input, d_model, d_inner, n_head, n_layer, dropout=0.1, activation="gelu", **kwargs):
        super(MotionDecoderV1, self).__init__()
        self.d_model = d_model
        self.position_enc = PositionalEncoding(d_model, max_len=1000)
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, 
                                                               nhead=n_head, 
                                                               dim_feedforward=d_inner, 
                                                               dropout=dropout, 
                                                               activation=activation)
        self.decoder = nn.TransformerDecoder(decoder_layer=transformer_decoder_layer, 
                                             num_layers=n_layer)
        
        self.linear = nn.Linear(d_model, d_input, bias=True)
        
    def forward(self, z, lengths):
        """
        :param z: [batch_size, 1, dim]
        :param latents: list of integer
        """
        batch_size = z.shape[0]
        device = z.device

        # Get mask
        mask = lengths_to_mask(lengths, device)
        
        # Switch z
        latent = z.permute(1, 0, 2) # [1, batch_size, dim]

        # Construct temporal queries
        time_queries = torch.zeros(batch_size, mask.shape[1], self.d_model, device=device).float()
        time_queries = self.position_enc(time_queries)
        time_queries = time_queries.permute(1, 0, 2)

        # Decode
        x_dec = self.decoder(tgt=time_queries, memory=latent, tgt_key_padding_mask=~mask)
        
        # Switch back
        x_dec = x_dec.permute(1, 0, 2)    # [batch_size, nframes, dim]

        # Decode back to output feature space
        final = self.linear(x_dec)
        
        # Zero for padded area
        final[~mask] = 0.0

        return final

if __name__ == "__main__":
    import yaml, json
    import numpy as np
    # from networks.ude.seqvq import Quantizer
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    with open("configs/ude/config_ude_exp4.yaml", "r") as f:
        conf = yaml.safe_load(f)
    
    print("=" * 20, "Build model", "=" * 20)
    model = MotionEncoderV1(**conf["model"]["perception"]["all"])
    model = model.to(device)
    print("=" * 20, "Load weights", "=" * 20)
    checkpoint = torch.load(
        "logs/perception/vae/exp1/pretrained-1027/checkpoints/MotionVAE_final.pth", 
        map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["encoder"], strict=True)
    model.eval()
    
    # data = np.load("../dataset/AIST++/aligned/gWA_sFM_cAll_d25_mWA4_ch05.npy", allow_pickle=True).item()
    # motion = torch.from_numpy(data["motion_smpl"][::2]).unsqueeze(dim=0).float().to(device)
    
    data = np.load("../dataset/BEAT_v0.2.1/aligned_20230331/2_scott_0_8_8.npy", allow_pickle=True).item()
    motion = torch.from_numpy(data["body"][::4]).unsqueeze(dim=0).float().to(device)
    # exit()
    
    motion_segments, lengths = [], []
    for i in range(0, motion.size(1), 100):
        inp_motion = motion[:, i:i+160]
        if inp_motion.size(1) != 160: continue
        motion_segments.append(inp_motion)
        lengths.append(160)
    motion_segments = torch.cat(motion_segments, dim=0)
    print('---> motion: ', motion_segments.shape)
    motion_pad = torch.zeros(motion_segments.size(0), 40, motion_segments.size(-1)).float().to(device)
    motion_segments = torch.cat([motion_segments, motion_pad], dim=1)
    mask = torch.ones(motion_segments.size(0), motion_segments.size(1)).to(device)
    for i, l in enumerate(lengths):
        mask[i, l:] = 0
                
    # Encode motion
    motion_embedding = model(motion_segments, mask.bool()).loc.squeeze(dim=1)
    print('---> embedding: ', motion_embedding.shape)
    np.save("speech_motion_embedding.npy", motion_embedding.data.cpu().numpy())
