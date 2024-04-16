import os, sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.ude.layers import *
from networks.utils.positional_encoding import PositionalEncoding
import importlib

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class UNet(nn.Module):
    def __init__(self, inp_channel, out_channel, channels):
        super(UNet, self).__init__()
        self.inp_layer = nn.Conv1d(inp_channel, channels[0], kernel_size=3, stride=1, padding=1)
        self.down_layer1 = nn.Sequential(
            nn.Conv1d(channels[0], channels[1], kernel_size=3, stride=2, padding=1), 
            nn.LeakyReLU(0.2, inplace=True), 
            ResBlock(channel=channels[1])
        )
        self.down_layer2 = nn.Sequential(
            nn.Conv1d(channels[1], channels[2], kernel_size=3, stride=2, padding=1), 
            nn.LeakyReLU(0.2, inplace=True), 
            ResBlock(channel=channels[2])
        )
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_layer1 = nn.Sequential(
            nn.Conv1d(channels[2], channels[1], kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.up_res_1 = ResBlock(channel=channels[1])
        self.up_layer2 = nn.Sequential(
            nn.Conv1d(channels[1], channels[0], kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out_layer = nn.Conv1d(channels[0], out_channel, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x0 = self.inp_layer(x)
        x1 = self.down_layer1(x0)
        x2 = self.down_layer2(x1)
        y2 = self.up(x2)
        y2 = self.up_layer1(y2) + x1
        y2 = self.up_res_1(y2)
        y1 = self.up(y2)
        y1 = self.up_layer2(y1) + x0
        y = self.out_layer(y1)
        return y

""" Quantizers """
class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, **kwargs):
        super(Quantizer, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding, there are two terms in the loss: 
        # 1) z_q - z.detach(): pull codebook embeddings to close to encoded motion embedding, and 
        # 2) z_q.detach() - z: pull encoded motion embeddings to close to codebook embeddings.
        loss = torch.mean((z_q - z.detach())**2) + self.beta * \
               torch.mean((z_q.detach() - z)**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean + 1e-10)))
        return loss, z_q, min_encoding_indices, perplexity

    def map2index(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices

    def get_z_to_code_distance(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        return d.reshape(z.shape[0], z.shape[1], -1)

    def get_codebook_entry(self, indices, is_onhot=False):
        """
        Get the token embeddings by given token indices (onehot distribution)
        :param indices: [B, seq_len] or [B, seq_len, n_dim]
        :return z_q(B, seq_len, e_dim):
        """
        if not is_onhot:
            index_flattened = indices.view(-1)
            z_q = self.embedding(index_flattened)
            z_q = z_q.view(indices.shape + (self.e_dim, )).contiguous()
        else:
            z_q = torch.matmul(indices, self.embedding.weight).contiguous()  # [B, seq_len, n_dim]
        return z_q

    def multinomial(self, indices, K=10, temperature=2):
        """Sample tokens according to the embedding distance.
        :param indices: [batch_size, seq_len]
        :param K: top-k
        :param temperature: temperature parameter controls the smoothness of pdist distribution. 
            large temperature coefficient smoothens the distribution.
        """
        # Get query embedding
        B, T = indices.shape
        z_emb = self.get_codebook_entry(indices)    # [B, T, C]
        # Calculate the pairwise distance between query embedding and all embedding
        q_emb = self.embedding.weight.clone()
        pdist = torch.cdist(z_emb.contiguous().view(B*T, -1), q_emb)                 # [B*T, N]        
        pdist = torch.exp(-pdist / temperature)
        # Select top-k
        _, topk_idx = torch.topk(pdist, k=K, dim=-1)
        topk_data = [pdist[i, topk_idx[i]] for i in range(topk_idx.size(0))]
        topk_data = torch.stack(topk_data, dim=0)
        # Sample 1 data from the selected top-K data (the returns are index)
        sampled_idx_ = torch.multinomial(topk_data, num_samples=1, replacement=False)
        # We get the corresponding indices in pre-topk data space
        sampled_idx = [topk_idx[i, sampled_idx_[i]] for i in range(sampled_idx_.size(0))]
        # We get the corresponding sampled data
        sampled_data = [topk_data[i, sampled_idx_[i]] for i in range(sampled_idx_.size(0))]
        sampled_idx = torch.stack(sampled_idx, dim=0)
        sampled_data = torch.stack(sampled_data, dim=0)
        
        sampled_idx = sampled_idx.reshape(B, T)
        sampled_data = sampled_data.reshape(B, T, -1)
        topk_data = topk_data.reshape(B, T, -1)
        
        return sampled_idx, sampled_data, topk_data

    def semantic_aware_sampling(self, indices, logits, K=10, temperature=2):
        """Sample tokens according to the embedding distance.
        :param indices: [batch_size, seq_len]
        :param logits: [batch_size, seq_len, num_dim]
        :param K: top-k
        :param temperature: temperature parameter controls the smoothness of pdist distribution. 
            large temperature coefficient smoothens the distribution.
        """
        # Get query embedding
        B, T = indices.shape
        z_emb = self.get_codebook_entry(indices)    # [B, T, C]
        # Calculate the pairwise distance between query embedding and all embedding
        q_emb = self.embedding.weight.clone()
        pdist = F.cosine_similarity(z_emb.contiguous().view(B*T, -1), q_emb, dim=-1).unsqueeze(dim=0)
        emb_prob = F.softmax(pdist / temperature, dim=-1)                           # [B*T, N]
        logit_prob = F.softmax(logits.view(B*T, -1).clone(), dim=-1)
        
        _, emb_topk_idx = torch.topk(emb_prob, k=K, dim=-1)
        _, logit_topk_idx = torch.topk(logits.view(B*T, -1), k=K, dim=-1)
        
        def topk2mask(inp, topk_idx):
            mask = torch.zeros_like(inp)
            for i in range(topk_idx.size(1)):
                mask[0, topk_idx[0, i]] = 1
            return mask.long()
        
        # emb_mask = topk2mask(emb_prob, emb_topk_idx)
        # logit_mask = topk2mask(emb_prob, logit_topk_idx)
        # mask = emb_mask * logit_mask
        # prob = logits.view(B*T, -1).clone().masked_fill(mask.ne(1), float("-inf"))
        # prob = F.softmax(prob / temperature, dim=-1)
        prob = emb_prob * logit_prob
        
        # import numpy as np
        # np.savetxt("emb_prob.txt", emb_prob[0].data.cpu().numpy(), fmt="%06f")
        # np.savetxt("logit_prob.txt", logit_prob[0].data.cpu().numpy(), fmt="%06f")
        # np.savetxt("prob.txt", prob[0].data.cpu().numpy(), fmt="%06f")
        
        sampled_idx = torch.multinomial(prob, num_samples=1, replacement=True)
        return sampled_idx.reshape(B, T), None, None
    
    def multinomial_prob(self, indices, K=10, temperature=2):
        """Convert the onehot probability to multinomial probability.
        :param indices: [batch_size, seq_len]
        :param K: top-k
        :param temperature: temperature parameter controls the smoothness of pdist distribution
        """
        device = indices.device
        # Get query embedding
        B, T = indices.shape
        z_emb = self.get_codebook_entry(indices)    # [B, T, C]
        # Calculate the pairwise distance between query embedding and all embedding
        q_emb = self.embedding.weight.clone()
        pdist = torch.cdist(z_emb.contiguous().view(B*T, -1), q_emb)                 # [B*T, N]
        prob = torch.exp(-pdist)
        # Select top-K (actually the smallest top-k)
        _, topk_idx = torch.topk(pdist, k=self.n_e-K, dim=-1)
        topk_mask = [torch.zeros(self.n_e).to(device).scatter_(dim=0, index=idx, src=torch.ones(self.n_e-K).to(device)) for idx in topk_idx]
        topk_mask = torch.stack(topk_mask, dim=0)
        topk_mask = topk_mask.to(device)
        prob = prob.masked_fill(topk_mask == 1, float('-inf'))
        # Convert to multinomial probability
        prob = F.softmax(prob.contiguous().view(B, T, -1) / temperature, dim=-1)
        return prob
    
""" VQEncoders """
class VQEncoderV1(nn.Module):
    def __init__(self, input_size, channels, n_down, **kwargs):
        super(VQEncoderV1, self).__init__()
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return outputs  # [bs, nframes / 4, 1024]

""" VQDecoders """
class VQDecoderV1(nn.Module):
    def __init__(self, input_size, channels, n_resblk, n_up, **kwargs):
        super(VQDecoderV1, self).__init__()
        assert len(channels) == n_up + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            layers += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs  # [bs, nframes, 263]

class VQDecoderV2(nn.Module):
    """Conv1D + Conv1D decoder. 
    1. The Conv1D decode the codebooks back to sub-local motion sequences. 
       Then we convert the sub-local motion sequence to sub-global motion sequence.
       We define the sub-local motion sequence as: we set the global translation of 
       the root joints as the offsets relative to previous frame. 
    2. Conv1D then maps the sub-local motion sequence to global motion sequence.
    """
    def __init__(self, input_size, channels, n_resblk, n_up, **kwargs):
        super(VQDecoderV2, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        assert len(channels) == n_up + 1
        
        # Build stage-one convs (codebook to sub-local)
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]
        
        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        for i in range(n_up):
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"), 
                nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1), 
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.convs = nn.Sequential(*layers)
        self.convs.apply(init_weight)
        
        # Build stage-two convs (sub-local to global)
        output_size = channels[-1]
        self.convs_2 = UNet(inp_channel=output_size*2, out_channel=output_size, channels=[256, 512, 1024])
        
    def forward(self, inputs, return_pose=False):
        """
        :param inputs: [batch_size, n_tokens, dim]
        """
        # Decode codebook to sub-local motion sequence
        self.sublocal_outputs = self.convs(inputs.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_size, seq_len, out_dim]
        # Convert sub-local to sub-global
        self.subglobal_inputs = self.convert_sublocal_to_subglobal(self.sublocal_outputs)   # [batch_size, seq_len, out_dim]
        # Convert sub-global to global
        global_outputs = self.convs_2(torch.cat([self.sublocal_outputs, self.subglobal_inputs], dim=-1).permute(0, 2, 1)).permute(0, 2, 1)
        return global_outputs
        
    def convert_sublocal_to_subglobal(self, inputs):
        """
        :param input: [batch_size, seq_len, dim]
        """
        sl_trans = inputs[..., :3]
        pose = inputs[..., 3:]
        
        batch_size = sl_trans.size(0)
        seq_len = sl_trans.size(1)
        
        sg_trans = [torch.zeros(batch_size, 3).float().to(inputs.device)]
        for i in range(1, seq_len):
            sg_trans.append(sg_trans[-1] + sl_trans[:, i])
        sg_trans = torch.stack(sg_trans, dim=1)
        return torch.cat([sg_trans, pose], dim=-1)

    def get_sublocal_outputs(self):
        return self.sublocal_outputs
    
    def get_subglobal_outputs(self):
        return self.subglobal_inputs

class VQDecoderV3(nn.Module):
    """Conv + Transformer decoder. 
    1. The Convs decode the codebooks back to sub-local motion sequences. 
       Then we convert the sub-local motion sequence to sub-global motion sequence.
       We define the sub-local motion sequence as: we set the global translation of 
       the root joints as the offsets relative to previous frame. 
    2. Transformers then maps the sub-local motion sequence to global motion sequence.
    """
    def __init__(self, input_size, channels, n_resblk, n_up, hidden_dims, num_layers, num_heads, dropout, activation="gelu", **kwargs):
        super(VQDecoderV3, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        assert len(channels) == n_up + 1
        
        # Build convs (codebook to sub-local)
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]
        
        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        for i in range(n_up):
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"), 
                nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1), 
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.convs = nn.Sequential(*layers)
        self.convs.apply(init_weight)
        
        # Build transformers (sub-local to global)
        output_size = channels[-1]
        self.s2g_linear = nn.Linear(output_size, input_size, bias=False)
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, 
                                                               nhead=num_heads, 
                                                               dim_feedforward=hidden_dims, 
                                                               dropout=dropout, 
                                                               activation=activation)
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, 
                                                 num_layers=num_layers)
        self.position_encoding = PositionalEncoding(d_model=input_size, dropout=dropout, max_len=5000)
        self.final = nn.Conv1d(input_size, output_size, kernel_size=3, stride=1, padding=1)
        self.s2g_linear.apply(init_weight)
        self.final.apply(init_weight)
        
    def forward(self, inputs, return_pose=False):
        """
        :param inputs: [batch_size, n_tokens, dim]
        """
        # Decode codebook to sub-local motion sequence
        self.sublocal_outputs = self.convs(inputs.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_size, seq_len, out_dim]
        # Convert sub-local to sub-global
        self.subglobal_inputs = self.convert_sublocal_to_subglobal(self.sublocal_outputs)
        # Convert sub-global to global
        global_outputs = self.convert_subglobal_to_global(self.subglobal_inputs)
        return global_outputs
        
    def convert_sublocal_to_subglobal(self, inputs):
        """
        :param input: [batch_size, seq_len, dim]
        """
        sl_trans = inputs[..., :3]
        pose = inputs[..., 3:]
        
        batch_size = sl_trans.size(0)
        seq_len = sl_trans.size(1)
        
        sg_trans = [torch.zeros(batch_size, 3).float().to(inputs.device)]
        for i in range(1, seq_len):
            sg_trans.append(sg_trans[-1] + sl_trans[:, i])
        sg_trans = torch.stack(sg_trans, dim=1)
        return torch.cat([sg_trans, pose], dim=-1)
        
    def convert_subglobal_to_global(self, inputs):
        x = self.s2g_linear(inputs)
        x = x.permute(1, 0, 2)
        x = self.position_encoding(x)
        x = self.transformer(x).permute(1, 0, 2)
        y = self.final(x.permute(0, 2, 1))
        return y.permute(0, 2, 1)
        
    def get_sublocal_outputs(self):
        return self.sublocal_outputs
    
    def get_subglobal_outputs(self):
        return self.subglobal_inputs
    
if __name__ == '__main__':

    import yaml
    with open("configs/vqvae/t2m/config_vqvae_exp1.yaml", "r") as f:
        conf = yaml.safe_load(f)
    
    encoder = VQEncoderV1(**conf["model"]["body"]["vq_encoder"])
    decoder = VQDecoderV2(**conf["model"]["body"]["vq_decoder"])
    quantizer = Quantizer(**conf["model"]["body"]["quantizer"])
    
    checkpoint = torch.load("logs/vqvae/t2m/exp1/pretrained-1028/checkpoints/best_1027.pth", map_location=torch.device("cpu"))
    quantizer.load_state_dict(checkpoint["body_quantizer"], strict=True)
    
    # x = torch.randn(2, 40, 75)
    # z = encoder(x)
    # embedding_loss, vq_latents, emb_indices, perplexity = quantizer(z)
    # glob_output = decoder(vq_latents)                 # Global poses
    # local_output = decoder.get_sublocal_outputs()     # Sub-optimal global poses
    
    input = torch.load("debug.pt")
    # torch.save({"idx": idx_out.squeeze(dim=-1)-3, "logit": raw_pred_logit[..., 3:].clone()}, "debug.pt")
    indices = input["idx"].cpu()
    logits = input["logit"].cpu()
    quantizer.semantic_aware_sampling(indices=indices, logits=logits, K=20, temperature=0.1)
    
    
    
    