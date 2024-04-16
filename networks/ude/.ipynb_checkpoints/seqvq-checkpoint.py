import os, sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.ude_v2.layers import *
from networks.utils.positional_encoding import PositionalEncoding
import importlib

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

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
        :param temperature: temperature parameter controls the smoothness of pdist distribution
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
    
class SoftQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, gamma=0.1, temperature=0.1, **kwargs):
        super(SoftQuantizer, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        
    def forward(self, z):
        """
        Inputs the output of encoder network z and maps it to a discrete one-hot vector.
        :param z: [batch_size, seq_len, ndim]
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)
        
        # B x V, pairwise distance between z and z_q, [n_z, n_token]
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        
        # Calculate the pulling force, each z will affects every z_q proportionally to 
        # the distance from z to z_q, the smaller the distance is, the stronger the 
        # pulling force is. 
        exp_d = torch.exp(-d / self.gamma)
        w = F.softmax(exp_d / self.temperature, dim=-1) # [n_z, n_token]
   
class DaibinQuantizer(nn.Module):
    def __init__(self, feature_dim, codebook_size, **kwargs):
        super(DaibinQuantizer, self).__init__()
        self.feature_dim = feature_dim
        self.codebook_size = codebook_size
        self.embedding = nn.Embedding(self.codebook_size, self.feature_dim)
        bnd = 1.0 / self.feature_dim
        self.embedding.weight.data.uniform_(-bnd, bnd)
        self.n_e = codebook_size
        self.reset_cnt()

    def forward(self, x):
        x_size = x.size()
        self.min_encoding_indices = self.quantize(x)
        x_q = torch.index_select(self.embedding.weight, 0, self.min_encoding_indices.view(-1)) 
        loss = F.l1_loss(x_q, x, reduction="mean")
        x_q = x_q.view(*x_size)
        
        return loss, x_q, None, None

    def map2index(self, z):
        z_size = z.size()
        z_flattened = z.contiguous().view(-1, self.feature_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t()) # (B, K)
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices

    def get_codebook_entry(self, indices):
        indices_flattened = indices.contiguous().view(-1)
        z_q = self.embedding(indices_flattened)
        z_q = z_q.view(indices.shape + (self.feature_dim,)).contiguous()
        return z_q

    def quantize(self, x):
        x_size = x.size()
        assert(x_size[-1] == self.feature_dim)
        x = x.view(-1, self.feature_dim)

        d = torch.sum(x**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(x, self.embedding.weight.t()) # (B, K)
        
        self.min_encoding_indices = torch.argmin(d, dim=1).long() # (B)
        self.min_encoding_indices = self.min_encoding_indices.view(*x_size[:-1])
        
        return self.min_encoding_indices

    def reset_cnt(self):
        self.cnts = np.zeros([self.codebook_size], dtype=np.int32)

    def reinit_codebook(self):
        zero_idxes = np.where(self.cnts == 0)[0]
        sorted_idxes = np.argsort(-self.cnts)
        for i, idx in enumerate(zero_idxes):
            self.embedding.weight.data[idx] = self.embedding.weight.data[sorted_idxes[sorted_idxes[i]]].clone()
        print(self.cnts[sorted_idxes])

    def update_cnt(self, x=None):
        if x is not None: self.quantize(x)
        hit = F.one_hot(self.min_encoding_indices, self.codebook_size) 
        while len(hit.size()) > 1:
            hit = hit.sum(0)
        hit = hit.detach().cpu().numpy()
        self.cnts += hit
         
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

class VQEncoderV3(nn.Module):
    """
    Conv + Transformer encoder.
    Add linear projection layer to map the latents from high dimension space to low dimension space
    """
    def __init__(self, input_size, channels, n_down, 
                 num_heads, hidden_dim, num_layers, 
                 dropout, activation, **kwargs):
        super(VQEncoderV3, self).__init__()
        assert len(channels) == n_down

        # Convolution layers
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
        self.conv_layers = nn.Sequential(*layers)

        # Transformer layers
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=channels[-1],
                                                               nhead=num_heads,
                                                               dim_feedforward=hidden_dim,
                                                               dropout=dropout,
                                                               activation=activation)
        self.transformer = nn.TransformerEncoder(transformer_encoder_layer, 
                                                 num_layers=num_layers)
        self.sequence_pos_encoding = PositionalEncoding(channels[-1], dropout)

        self.conv_layers.apply(init_weight)
        self.transformer.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)    # [batch_size, dim, nframes]
        outputs = self.conv_layers(inputs).permute(2, 0, 1) # [nframes, batch_size, dim]
        outputs = self.sequence_pos_encoding(outputs)
        outputs = self.transformer(outputs).permute(1, 0, 2) # [batch_size, nframes, dim]
        return outputs

class DaibinEncoder(nn.Module):
    """VQ-Encoder from DaiBin."""
    def __init__(self, feature_dim, level, layer_per_level, base_dim, **kwargs):
        super(DaibinEncoder, self).__init__()
        with_bn = False

        last_dim = feature_dim 
        self.network = nn.Sequential()
        for level_id in range(level + 1):
            dim = base_dim * (2 ** level_id)
            for layer_id in range(layer_per_level):
                stride = 2 if (level_id > 0 and layer_id == 0) else 1
                layer = nn.Conv1d(last_dim, dim, 3, stride, 1)
                layer_name = f'{level_id+1}_{layer_id+1}'
                self.network.add_module(f'conv{layer_name}', layer)
                last_dim = dim 

            if level_id != level or layer_id != layer_per_level - 1:
                if with_bn:
                    self.network.add_module(f'bn{layer_name}', nn.BatchNorm1d(dim))
                self.network.add_module(f'relu{layer_name}', nn.ReLU(inplace=True))

        latent_dim = None
        if latent_dim is None:
            self.latent_dim = base_dim * (2 ** level)
        else:
            self.latent_dim = latent_dim
            self.network.add_module('conv_last', nn.Conv1d(last_dim, latent_dim, 1, 1, 0))

    def forward(self, x):
        # x: (B, T, D)
        return self.network(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
    
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
    """Conv + Transformer decoder.
    """
    def __init__(self, input_size, channels, n_resblk, n_up, hidden_dims, num_layers, num_heads, dropout, activation="gelu", **kwargs):
        super(VQDecoderV2, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        assert len(channels) == n_up + 1
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
                nn.Conv1d(channels[i], channels[i], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        self.convs = nn.Sequential(*layers)
        self.convs.apply(init_weight)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=channels[-2],
                                                               nhead=num_heads,
                                                               dim_feedforward=hidden_dims,
                                                               dropout=dropout,
                                                               activation=activation)
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, 
                                                 num_layers=num_layers)
        self.transformer.apply(init_weight)
        self.sequence_pos_encoding = PositionalEncoding(channels[-2], dropout)

        self.linear = nn.Linear(channels[-2], channels[-1])

    def forward(self, inputs, noise=None):
        x = inputs.permute(0, 2, 1)                 # [batch_size, num_dims, num_frames]
        x = self.convs(x).permute(2, 0, 1)          # [num_frames, batch_size, num_dims]
        x = self.sequence_pos_encoding(x)
        x = self.transformer(x)
        outputs = self.linear(x).permute(1, 0, 2)   # [batcn_size, num_frames, num_dims]
        return outputs  # [bs, nframes, 263]

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
        subglobal_inputs = self.convert_sublocal_to_subglobal(self.sublocal_outputs)
        # Convert sub-global to global
        global_outputs = self.convert_subglobal_to_global(subglobal_inputs)
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
    
class DaibinDecoder(nn.Module):
    def __init__(self, feature_dim, level, layer_per_level, base_dim, **kwargs):
        super(DaibinDecoder, self).__init__()
        with_bn = False
        
        last_dim = base_dim * (2 ** level)
        self.network = nn.Sequential()
        latent_dim = None
        if latent_dim is not None:
            self.network.add_module('conv_first', nn.Conv1d(latent_dim, last_dim, 1, 1, 0))

        for level_id in range(level + 1):
            dim = base_dim * (2 ** (level - level_id))
            for layer_id in range(layer_per_level):
                is_last_layer = (level_id == level and layer_id == layer_per_level - 1)
                if is_last_layer:
                    dim = feature_dim

                if level_id > 0 and layer_id == 0:
                    # upsample = nn.ConvTranspose1d(last_dim, last_dim, 3, stride=2, padding=1, output_padding=1)
                    upsample = nn.Upsample(scale_factor=2, mode='nearest')
                    self.network.add_module(f'upsample{level_id}', upsample)

                layer = nn.Conv1d(last_dim, dim, 3, 1, 'same')
                layer_name = f'{level_id+1}_{layer_id+1}'
                self.network.add_module(f'conv{layer_name}', layer)
                last_dim = dim 

                if not is_last_layer:
                    if with_bn:
                        self.network.add_module(f'bn{layer_name}', nn.BatchNorm1d(dim))
                    self.network.add_module(f'relu{layer_name}', nn.ReLU(inplace=True))
                    
    def forward(self, x):
        return self.network(x.permute(0, 2, 1)).permute(0, 2, 1)

class VQDecoderVelocityV1(VQDecoderV1):
    def __init__(self, input_size, channels, n_resblk, n_up, activation, vel_decoder, **kwargs):
        super(VQDecoderVelocityV1, self).__init__(input_size, channels, n_resblk, n_up)
        assert isinstance(vel_decoder, dict)
        # We freeze the parameters for gradient computation
        for p in super().parameters():
            p.requires_grad = False 
            
        # Estimate velocity
        self.vel_decoder = importlib.import_module(
            vel_decoder["arch_path"], package="networks").__getattribute__(
                vel_decoder["arch_name"])(**vel_decoder)
        
    def forward(self, inputs, noise=None, return_pose=False):
        # We fixed the pose decoding part, so we don't allow gradient computation at this stage.
        with torch.no_grad():
            pose_outputs = super().forward(inputs)
        
        # Predict the velocity of this pose
        vel_outputs = self.vel_decoder(pose_outputs)
        
        if return_pose:
            return pose_outputs, vel_outputs
        else:
            return vel_outputs
    
    def parameters(self):
        return filter(lambda p: p.requires_grad, super().parameters())

class VQDecoderVelocityV2(VQDecoderV2):
    def __init__(self, input_size, channels, n_resblk, n_up, 
                 hidden_dims, num_layers, num_heads, dropout, 
                 activation, vel_decoder, **kwargs):
        super(VQDecoderVelocityV2, self).__init__(
            input_size, channels, n_resblk, n_up, hidden_dims, num_layers, num_heads, dropout, activation, **kwargs)
        assert isinstance(vel_decoder, dict)
        # We freeze the parameters for gradient computation
        for p in super().parameters():
            p.requires_grad = False 
        
        # Estimate velocity
        self.vel_decoder = importlib.import_module(
            vel_decoder["arch_path"], package="networks").__getattribute__(
                vel_decoder["arch_name"])(**vel_decoder)
        
    def forward(self, inputs, noise=None, return_pose=False):
        # We fixed the pose decoding part, so we don't allow gradient computation at this stage.
        with torch.no_grad():
            pose_outputs = super().forward(inputs)
        
        # Predict the velocity of this pose
        vel_outputs = self.vel_decoder(pose_outputs)
        
        if return_pose:
            return pose_outputs, vel_outputs
        else:
            return vel_outputs
    
    def parameters(self):
        return filter(lambda p: p.requires_grad, super().parameters())

""" Velocity Decoders """
class VelDecoderV1(nn.Module):
    """ Transformer-based translation decoder """
    def __init__(
        self, d_input, d_model, d_inner, n_heads, n_layers, 
        dropout=0.1, activation="gelu", 
        causal_attention=False, **kwargs
    ):
        super(VelDecoderV1, self).__init__()
        self.causal_attention = causal_attention
        self.pose_linear = nn.Sequential(
            nn.Linear(d_input, d_model), 
            nn.LayerNorm(d_model) 
        )
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                               nhead=n_heads,
                                                               dim_feedforward=d_inner,
                                                               dropout=dropout,
                                                               activation=activation)
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, 
                                                 num_layers=n_layers)
        self.vel_linear = nn.Linear(d_model, 3)
        self.sequence_pos_encoding = PositionalEncoding(d_model, dropout)
        
    def forward(self, pose_inputs):
        t = pose_inputs.size(1)
        out = self.pose_linear(pose_inputs)
        out = out.permute(1, 0, 2)          # [B, T, C] -> [T, B, C]
        out = self.sequence_pos_encoding(out)
        if self.causal_attention:
            mask = self.gen_causal_mask(seq_len=t, device=pose_inputs.device)
            out = self.transformer(out, mask=~mask)
        else:
            out = self.transformer(out, mask=None)
        out = out.permute(1, 0, 2)          # [T, B, C] -> [B, T, C]
        out = self.vel_linear(out)          # [B, T, C] -> [B, T, 3]
        return out
    
    def gen_causal_mask(self, seq_len, device):
        """If a BoolTensor is provided, positions with ''True'' is not allowed to attend 
        while ''False'' values will be unchanged.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device)==1).transpose(0, 1)
        return mask

class VelDecoderV2(nn.Module):
    def __init__(
        self, d_input, d_model, d_inner, n_layers, 
        dropout=0.1, bidirectional=False, **kwargs
    ):
        super(VelDecoderV2, self).__init__()
        self.bidirectional = bidirectional
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.init_linear = nn.Linear(d_input, d_model)
        self.pose_linear = nn.Sequential(
            nn.Linear(d_input, d_model), 
            nn.LayerNorm(d_model)
        )
        self.lstm = nn.LSTM(input_size=d_model, 
                            hidden_size=d_inner, 
                            num_layers=n_layers, 
                            dropout=dropout, 
                            bidirectional=bidirectional, 
                            batch_first=False)
        self.vel_linear = nn.Linear(d_inner, 3)
    
    def forward(self, pose_inputs):
        """
        :param pose_inputs: [B, T, D]
        """
        B = pose_inputs.size(0)
        t = pose_inputs.size(1)
        
        D = 2 if self.bidirectional else 1
        h0 = torch.zeros(D * self.n_layers, B, self.d_inner).float().to(pose_inputs.device)
        c0 = torch.zeros(D * self.n_layers, B, self.d_inner).float().to(pose_inputs.device)
        inp = self.pose_linear(pose_inputs)
        out, (hn, cn) = self.lstm(inp.permute(1, 0, 2), (h0, c0))
        out = out.permute(1, 0, 2)      # [T, B, C] -> [B, T, C]
        out = self.vel_linear(out)      # [B, T, C] -> [B, T, 3]
        return out
        
class VelDecoderV3(nn.Module):
    """ Transformer-based translation decoder """
    def __init__(
        self, d_input, d_model, d_inner, n_heads, 
        enc_n_layers, dec_n_layers,  
        dropout=0.1, activation="gelu", 
        causal_attention=False, **kwargs
    ):
        super(VelDecoderV3, self).__init__()
        self.causal_attention = causal_attention
        self.pose_linear = nn.Sequential(
            nn.Linear(d_input, d_model), 
            nn.LayerNorm(d_model) 
        )
        enc_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                                   nhead=n_heads,
                                                                   dim_feedforward=d_inner,
                                                                   dropout=dropout,
                                                                   activation=activation)
        self.enc_transformer = nn.TransformerEncoder(encoder_layer=enc_transformer_encoder_layer, 
                                                     num_layers=enc_n_layers)
        
        dec_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                                   nhead=n_heads,
                                                                   dim_feedforward=d_inner,
                                                                   dropout=dropout,
                                                                   activation=activation)
        self.dec_transformer = nn.TransformerEncoder(encoder_layer=dec_transformer_encoder_layer, 
                                                     num_layers=dec_n_layers)
        self.vel_linear = nn.Linear(d_model, 3)
        self.sequence_pos_encoding = PositionalEncoding(d_model, dropout)
        
    def forward(self, pose_inputs):
        """We assume the translation could be predicted from the residual between two consecutive frames.
        """
        pose_residual = pose_inputs[:, 1:] - pose_inputs[:, :-1]    # [B, T-1, C]
        t = pose_residual.size(1)
        out = self.pose_linear(pose_residual)
        out = out.permute(1, 0, 2)          # [B, T-1, C] -> [T-1, B, C]
        out = self.sequence_pos_encoding(out)
        out = self.enc_transformer(out)     # Encode the global context information
        if self.causal_attention:
            mask = self.gen_causal_mask(seq_len=t, device=pose_inputs.device)
            out = self.dec_transformer(out, mask=~mask)
        else:
            out = self.dec_transformer(out, mask=None)
        out = out.permute(1, 0, 2)          # [T-1, B, C] -> [B, T-1, C]
        out = self.vel_linear(out)          # [B, T-1, C] -> [B, T-1, 3]
        init_root = torch.zeros_like(out[:, :1])
        out = torch.cat([init_root, out], dim=1)
        return out
    
    def gen_causal_mask(self, seq_len, device):
        """If a BoolTensor is provided, positions with ''True'' is not allowed to attend 
        while ''False'' values will be unchanged.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device)==1).transpose(0, 1)
        return mask

""" Velocity and Orientation Decoders """
class VelOrienDecoderV1(nn.Module):
    """ Transformer-based translation decoder """
    def __init__(
        self, d_input, d_model, d_inner, n_heads, n_layers, 
        dropout=0.1, activation="gelu", 
        causal_attention=False, **kwargs
    ):
        super(VelOrienDecoderV1, self).__init__()
        self.causal_attention = causal_attention
        self.pose_linear = nn.Sequential(
            nn.Linear(d_input, d_model), 
            nn.LayerNorm(d_model) 
        )
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                               nhead=n_heads,
                                                               dim_feedforward=d_inner,
                                                               dropout=dropout,
                                                               activation=activation)
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, 
                                                 num_layers=n_layers)
        self.vel_linear = nn.Linear(d_model, 6)
        self.sequence_pos_encoding = PositionalEncoding(d_model, dropout)
        
    def forward(self, pose_inputs):
        t = pose_inputs.size(1)
        out = self.pose_linear(pose_inputs)
        out = out.permute(1, 0, 2)          # [B, T, C] -> [T, B, C]
        out = self.sequence_pos_encoding(out)
        if self.causal_attention:
            mask = self.gen_causal_mask(seq_len=t, device=pose_inputs.device)
            out = self.transformer(out, mask=~mask)
        else:
            out = self.transformer(out, mask=None)
        out = out.permute(1, 0, 2)          # [T, B, C] -> [B, T, C]
        out = self.vel_linear(out)          # [B, T, C] -> [B, T, 3]
        return out
    
    def gen_causal_mask(self, seq_len, device):
        """If a BoolTensor is provided, positions with ''True'' is not allowed to attend 
        while ''False'' values will be unchanged.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device)==1).transpose(0, 1)
        return mask

class VelOrienDecoderV2(nn.Module):
    def __init__(
        self, d_input, d_model, d_inner, n_layers, 
        dropout=0.1, bidirectional=False, **kwargs
    ):
        super(VelOrienDecoderV2, self).__init__()
        self.bidirectional = bidirectional
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.init_linear = nn.Linear(d_input, d_model)
        self.pose_linear = nn.Sequential(
            nn.Linear(d_input, d_model), 
            nn.LayerNorm(d_model)
        )
        self.lstm = nn.LSTM(input_size=d_model, 
                            hidden_size=d_inner, 
                            num_layers=n_layers, 
                            dropout=dropout, 
                            bidirectional=bidirectional, 
                            batch_first=False)
        self.vel_linear = nn.Linear(d_inner, 6)
    
    def forward(self, pose_inputs):
        """
        :param pose_inputs: [B, T, D]
        """
        B = pose_inputs.size(0)
        t = pose_inputs.size(1)
        
        D = 2 if self.bidirectional else 1
        h0 = torch.zeros(D * self.n_layers, B, self.d_inner).float().to(pose_inputs.device)
        c0 = torch.zeros(D * self.n_layers, B, self.d_inner).float().to(pose_inputs.device)
        inp = self.pose_linear(pose_inputs)
        out, (hn, cn) = self.lstm(inp.permute(1, 0, 2), (h0, c0))
        out = out.permute(1, 0, 2)      # [T, B, C] -> [B, T, C]
        out = self.vel_linear(out)      # [B, T, C] -> [B, T, 3]
        return out

class VelOrienDecoderV3(nn.Module):
    """ Transformer-based translation decoder """
    def __init__(
        self, d_input, d_model, d_inner, n_heads, 
        enc_n_layers, dec_n_layers,  
        dropout=0.1, activation="gelu", 
        causal_attention=False, **kwargs
    ):
        super(VelDecoderV3, self).__init__()
        self.causal_attention = causal_attention
        self.pose_linear = nn.Sequential(
            nn.Linear(d_input, d_model), 
            nn.LayerNorm(d_model) 
        )
        enc_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                                   nhead=n_heads,
                                                                   dim_feedforward=d_inner,
                                                                   dropout=dropout,
                                                                   activation=activation)
        self.enc_transformer = nn.TransformerEncoder(encoder_layer=enc_transformer_encoder_layer, 
                                                     num_layers=enc_n_layers)
        
        dec_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                                   nhead=n_heads,
                                                                   dim_feedforward=d_inner,
                                                                   dropout=dropout,
                                                                   activation=activation)
        self.dec_transformer = nn.TransformerEncoder(encoder_layer=dec_transformer_encoder_layer, 
                                                     num_layers=dec_n_layers)
        self.vel_linear = nn.Linear(d_model, 6)
        self.sequence_pos_encoding = PositionalEncoding(d_model, dropout)
        
    def forward(self, pose_inputs):
        """We assume the translation could be predicted from the residual between two consecutive frames.
        """
        pose_residual = pose_inputs[:, 1:] - pose_inputs[:, :-1]    # [B, T-1, C]
        t = pose_residual.size(1)
        out = self.pose_linear(pose_residual)
        out = out.permute(1, 0, 2)          # [B, T-1, C] -> [T-1, B, C]
        out = self.sequence_pos_encoding(out)
        out = self.enc_transformer(out)     # Encode the global context information
        if self.causal_attention:
            mask = self.gen_causal_mask(seq_len=t, device=pose_inputs.device)
            out = self.dec_transformer(out, mask=~mask)
        else:
            out = self.dec_transformer(out, mask=None)
        out = out.permute(1, 0, 2)          # [T-1, B, C] -> [B, T-1, C]
        out = self.vel_linear(out)          # [B, T-1, C] -> [B, T-1, 3]
        init_root = torch.zeros_like(out[:, :1])
        out = torch.cat([init_root, out], dim=1)
        return out
    
    def gen_causal_mask(self, seq_len, device):
        """If a BoolTensor is provided, positions with ''True'' is not allowed to attend 
        while ''False'' values will be unchanged.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device)==1).transpose(0, 1)
        return mask

if __name__ == '__main__':

    # conf = {
    #     "arch_path": '.ude_v2.seqvq', "arch_name": 'VQDecoderV1', 
    #     "input_size": 512, "channels": [1024, 1024, 75], 
    #     "n_resblk": 3, "n_up": 2, "hidden_dims": 2048, 
    #     "num_layers": 2, "num_heads": 4, "dropout": 0.1, "activation": "gelu", 
    #     "vel_decoder": {
    #         "arch_path": '.ude_v2.seqvq', 
    #         "arch_name": 'VelDecoderV3', 
    #         "d_input": 75, 
    #         "d_model": 256, 
    #         "d_inner": 768, 
    #         "n_heads": 4, 
    #         "enc_n_layers": 2, 
    #         "dec_n_layers": 1, 
    #         "dropout": 0.1, 
    #         "activation": "gelu",
    #         "causal_attention": True
    #     }
    #     # "vel_decoder": {
    #     #     "arch_path": '.ude_v2.seqvq', 
    #     #     "arch_name": 'VelDecoderV2', 
    #     #     "d_input": 75, 
    #     #     "d_model": 128, 
    #     #     "d_inner": 512, 
    #     #     "n_layers": 2, 
    #     #     "bidirectional": False, 
    #     #     "dropout": 0.1, 
    #     #     "activation": "gelu",
    #     # }
    # }

    # Decoder = VQDecoderVelocityV1(**conf)
    # parameters = Decoder.parameters()
    # # print(len(list(parameters)))
    # x = torch.randn(4, 16, 512)
    # y = Decoder(x)
    # print(y.shape)
    
    enc_conf = {
        "feature_dim": 75, 
        "level": 2, 
        "layer_per_level": 1, 
        "base_dim": 128
    }
    quant_conf = {
        "feature_dim": 512, 
        "codebook_size": 2048
    }
    dec_conf = {
        "feature_dim": 75, 
        "level": 2, 
        "layer_per_level": 2, 
        "base_dim": 128
    }
    
    encoder = DaibinEncoder(**enc_conf)
    decoder = DaibinDecoder(**dec_conf)
    quantizer = DaibinQuantizer(**quant_conf)
    # print(encoder)
    # print(quantizer)
    # print(decoder)
    
    checkpoint = torch.load("from_daibin/exp312/vqvae_last.pth", map_location=torch.device("cpu"))
    enc_state_dict = {}
    dec_state_dict = {}
    quant_state_dict = {}
    for key, val in checkpoint.items():
        if "encoder" in key:
            enc_state_dict[key.replace("encoder.", "")] = val
        elif "decoder" in key:
            dec_state_dict[key.replace("decoder.", "")] = val
        elif "quantizer" in key:
            quant_state_dict[key.replace("quantizer.", "")] = val
    encoder.load_state_dict(enc_state_dict, strict=True)
    decoder.load_state_dict(dec_state_dict, strict=True)
    quantizer.load_state_dict(quant_state_dict, strict=True)
    
    
    
    
    