import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import torchaudio
from torchaudio import transforms


class Res2DMaxPoolModule(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2):
        super(Res2DMaxPoolModule, self).__init__()
        self.conv_1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

        # residual
        self.diff = False
        if input_channels != output_channels:
            self.conv_3 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True

    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.mp(self.relu(out))
        return out
# Transformer modules
"""
    Referenced PyTorch implementation of Vision Transformer by Lucidrains.
    https://github.com/lucidrains/vit-pytorch.git
"""
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                            )
                        ),
                        Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

class TFRep(nn.Module):
    """
    time-frequency represntation
    """
    def __init__(self, 
                sample_rate= 16000,
                f_min=0,
                f_max=8000,
                n_fft=1024,
                win_length=1024,
                hop_length = int(0.01 * 16000),
                n_mels = 128,
                power = None,
                pad= 0,
                normalized= False,
                center= True,
                pad_mode= "reflect"
                ):
        super(TFRep, self).__init__()
        self.window = torch.hann_window(win_length)
        self.spec_fn = torchaudio.transforms.Spectrogram(
            n_fft = n_fft,              # 1024
            win_length = win_length,    # 1024
            hop_length = hop_length,    # 160
            power = power               # none
        )
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels,             # 128
            sample_rate,        # 160000
            f_min,              # 0
            f_max,              # 8000
            n_fft // 2 + 1)
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def melspec(self, wav):
        spec = self.spec_fn(wav)
        power_spec = spec.real.abs().pow(2)
        # power_spec = spec.abs().pow(2)
        mel_spec = self.mel_scale(power_spec)
        mel_spec = self.amplitude_to_db(mel_spec)
        return mel_spec

    def spec(self, wav):
        spec = self.spec_fn(wav)
        real = spec.real
        imag = spec.imag
        power_spec = real.abs().pow(2)
        eps = 1e-10
        mag = torch.clamp(mag ** 2 + phase ** 2, eps, np.inf) ** 0.5
        cos = real / mag
        sin = imag / mag
        return power_spec, imag, mag, cos, sin

class Res2DMaxPoolModule(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2):
        super(Res2DMaxPoolModule, self).__init__()
        self.conv_1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

        # residual
        self.diff = False
        if input_channels != output_channels:
            self.conv_3 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True

    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.mp(self.relu(out))
        return out
    
class ResFrontEnd(nn.Module):
    """
    After the convolution layers, we flatten the time-frequency representation to be a vector.
    mix_type : cf -> mix channel and frequency dim
    mix_type : ft -> mix frequency and time dim
    """
    def __init__(self, input_size ,conv_ndim, attention_ndim, mix_type="cf",nharmonics=1):
        super(ResFrontEnd, self).__init__()
        self.mix_type = mix_type
        self.input_bn = nn.BatchNorm2d(nharmonics)
        self.layer1 = Res2DMaxPoolModule(nharmonics, conv_ndim, pooling=(2, 2))
        self.layer2 = Res2DMaxPoolModule(conv_ndim, conv_ndim, pooling=(2, 2))
        self.layer3 = Res2DMaxPoolModule(conv_ndim, conv_ndim, pooling=(2, 2))
        self.layer4 = Res2DMaxPoolModule(conv_ndim, conv_ndim, pooling=(2, 2))
        F,T = input_size
        self.ntime = T // 2 // 2 // 2 // 2
        self.nfreq = F // 2 // 2 // 2 // 2
        if self.mix_type == "ft":
            self.fc_ndim = conv_ndim
        else:
            self.fc_ndim = self.nfreq * conv_ndim
        self.fc = nn.Linear(self.fc_ndim, attention_ndim)

    def forward(self, hcqt):
        # batch normalization
        out = self.input_bn(hcqt)
        # CNN
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # permute and channel control
        b, c, f, t = out.shape
        if self.mix_type == "ft":
            out = out.contiguous().view(b, c, -1)  # batch, channel, tf_dim
            out = out.permute(0,2,1) # batch x length x dim
        else:
            out = out.permute(0, 3, 1, 2)  # batch, time, conv_ndim, freq
            out = out.contiguous().view(b, t, -1)  # batch, length, hidden
        out = self.fc(out)  # batch, time, attention_ndim
        return out
    
class MusicTransformer(nn.Module):
    def __init__(self,
                audio_representation,
                frontend, 
                audio_rep,
                is_vq=False, 
                dropout=0.1, 
                attention_ndim=256,
                attention_nheads=8,
                attention_nlayers=4,
                attention_max_len=512
        ):
        super(MusicTransformer, self).__init__()
        # Input preprocessing
        self.audio_representation = audio_representation
        self.audio_rep = audio_rep
        # Input embedding
        self.frontend = frontend
        self.is_vq = is_vq
        self.vq_modules = None
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, attention_max_len + 1, attention_ndim))
        self.cls_token = nn.Parameter(torch.randn(attention_ndim))
        # transformer
        self.transformer = Transformer(
            attention_ndim,
            attention_nlayers,
            attention_nheads,
            attention_ndim // attention_nheads,
            attention_ndim * 4,
            dropout,
        )
        self.to_latent = nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.audio_rep == "mel":
            spec = self.audio_representation.melspec(x)
            spec = spec.unsqueeze(1)
        elif self.audio_rep == "stft":
            spec = None
        h_audio = self.frontend(spec) # B x L x D
        if self.is_vq:
            h_audio = self.vq_modules(h_audio)
        cls_token = self.cls_token.repeat(h_audio.shape[0], 1, 1)
        h_audio = torch.cat((cls_token, h_audio), dim=1)
        h_audio += self.pos_embedding[:, : h_audio.size(1)]
        h_audio = self.dropout(h_audio)
        z_audio = self.transformer(h_audio)
        return z_audio
    
class MTR(nn.Module):
    def __init__(self, conf):
        super(MTR, self).__init__()
        self.conf = conf
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        audio_preprocessor = TFRep(
            sample_rate=conf["sr"], 
            f_min=0, 
            f_max=int(conf["sr"] / 2), 
            n_fft=conf["n_fft"], 
            win_length=conf["win_length"], 
            hop_length=int(0.01 * conf["sr"]), 
            n_mels=conf["mel_dim"]
        )
        frontend = ResFrontEnd(
            input_size=(conf["mel_dim"], int(100 * conf["duration"]) + 1), 
            conv_ndim=128, 
            attention_ndim=conf["attention_ndim"], 
            mix_type=conf["mix_type"]
        )
        self.model = MusicTransformer(
            audio_representation=audio_preprocessor, 
            frontend=frontend, 
            audio_rep=conf["audio_rep"], 
            attention_nlayers=conf["attention_nlayers"], 
            attention_ndim=conf["attention_ndim"]
        )
        # self.audio_projector = nn.Sequential(nn.LayerNorm(conf["attention_ndim"]), nn.Linear(audio_dim, mlp_dim, bias=False))

        self.eval()
        self.to(self.device)
        self.from_pretrained()
        
    def eval(self):
        self.model.eval()
    
    def to(self, device):
        self.model = self.model.to(device)
    
    def freeze(self):
        """Freeze the parameters to make them untrainable.
        """
        for p in self.model.parameters():
            p.requires_grad = False
            
    def from_pretrained(self):
        checkpoints = torch.load(self.conf["model"], map_location=torch.device("cpu"))
        state_dict = {}
        for name, val in checkpoints["state_dict"].items():
            if name.startswith('module.'):
                new_name = name[len('module.'):]
            if "audio_encoder" in new_name:
                new_name = new_name.replace("audio_encoder", "model")
                state_dict[new_name] = val
        super().load_state_dict(state_dict, strict=True)
        print("loaded from pretrained {:s}".format(self.conf["model"]))
    
    def forward(self, audios, mask=None):
        """
        :param audios: [batch_size, seq_len] mel spectrum
        :param mask: (optional)
        """
        with torch.no_grad():
            audio_emb = self.model(audios)
            mask = torch.ones(audio_emb.size(0), audio_emb.size(1)).bool().to(self.device)
        return audio_emb, mask
    
    def encode_audio(self, audios, mask=None):
        with torch.no_grad():
            audio_emb = self.model(audios)
            # h_audio = audio_emb[:, 0, :]
            h_audio = audio_emb.mean(dim=1)
        return h_audio

if __name__ == "__main__":
    
    waveform, sr = torchaudio.load("../dataset/AIST++/wav/mBR0.wav")
    # waveform = np.random.
    waveform = torch.randn(2, 85333)
    # print('sample rate: ', sr)
    # spectrogram_transform = transforms.Spectrogram(n_fft=1024, win_length=1024, hop_length=160, power=2)
    # spectrogram = spectrogram_transform(waveform)
    # print(waveform.shape, spectrogram.shape)
    # melscale_transform = transforms.MelScale(128, sample_rate=sr, f_min=0, f_max=8000, n_stft=1024 // 2 + 1)
    # melscale_spectrogram = melscale_transform(spectrogram)
    # print(waveform.shape, melscale_spectrogram.shape)
    
    conf = {
        "model": "networks/ude_v2/pretrained-model/music-text-representation/model/best.pth", 
        "sr": 16000, 
        "n_fft": 1024, 
        "win_length": 1024, 
        "mel_dim": 128, 
        "duration": 9.91, 
        "attention_ndim": 256, 
        "mix_type": "cf", 
        "audio_rep": "mel", 
        "attention_nlayers": 4, 
        "attention_ndim": 256
    }
    model = MTR(conf)
    # model.from_pretrained()
    model.freeze()
    for name, val in model.state_dict().items():
        print(name, val.min().item(), val.max().item())
    embeds, mask = model(waveform.float().to(model.device), mask=None)
    print(waveform.shape, embeds.shape, mask.shape)