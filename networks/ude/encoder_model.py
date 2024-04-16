import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class AudioModel(nn.Module):
    def __init__(self, conf):
        super(AudioModel, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.input = nn.Linear(conf["d_input"], conf["d_model"])
        self.position_enc = PositionalEncoding(d_model=conf["d_model"], max_len=5000)
        transformer_encoder = nn.TransformerEncoderLayer(d_model=conf["d_model"], 
                                                         nhead=conf["n_head"], 
                                                         dim_feedforward=conf["d_inner"], 
                                                         dropout=conf["dropout"], 
                                                         activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_encoder, 
                                                 num_layers=conf["n_layers"])
        
        self.output = nn.Sequential(
            nn.LayerNorm(conf["d_model"]), 
            nn.Linear(conf["d_model"], conf["d_model"])
        )
        
    def forward(self, x, mask):
        x = self.input(x)
        x = self.position_enc(x)
        x = self.transformer(x.permute(1, 0, 2), src_key_padding_mask=~mask)
        x = self.output(x.permute(1, 0, 2))
        return x, mask
    
    def freeze(self):
        """Freeze the parameters to make them untrainable.
        """
        pass
    
    def from_pretrained(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))["model"]
        state_dict = {}
        for name, val in checkpoint.items():
            if "pretrained_models.audio_encoder" in name:
                state_dict[name.replace("pretrained_models.audio_encoder.", "")] = val
        super().load_state_dict(state_dict, strict=True)
        print("loaded from pretrained {:s}".format(ckpt_path))
        
    def encode_audio(self, x, mask=None):
        with torch.no_grad():
            x = self.input(x)
            x = self.position_enc(x)
            if mask is not None:
                x = self.transformer(x.permute(1, 0, 2), src_key_padding_mask=~mask)
            else:
                x = self.transformer(x.permute(1, 0, 2))
            x = self.output(x.permute(1, 0, 2))
        return x.mean(dim=1)
    
class SpeechModel(AudioModel):
    def __init__(self, conf):
        super(SpeechModel, self).__init__(conf)
    
    def forward(self, x, mask):
        return super().forward(x, mask)
    
class WordModel(nn.Module):
    def __init__(self, conf):
        super(WordModel, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.embed = nn.Embedding(num_embeddings=conf["n_tokens"], embedding_dim=conf["d_model"])
        self.position_enc = PositionalEncoding(d_model=conf["d_model"], max_len=5000)
        transformer_encoder = nn.TransformerEncoderLayer(d_model=conf["d_model"], 
                                                         nhead=conf["n_head"], 
                                                         dim_feedforward=conf["d_inner"], 
                                                         dropout=conf["dropout"], 
                                                         activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_encoder, 
                                                 num_layers=conf["n_layers"])
        
        self.output = nn.Sequential(
            nn.LayerNorm(conf["d_model"]), 
            nn.Linear(conf["d_model"], conf["d_model"])
        )

    def forward(self, x, mask=None):
        emb = self.embed(x)
        emb = self.position_enc(emb)
        emb = self.transformer(emb.permute(1, 0, 2), src_key_padding_mask=~mask)
        emb = self.output(emb.permute(1, 0, 2))
        mask = torch.ones(emb.size(0), emb.size(1)).bool().to(self.device)
        return emb, mask
    
    def freeze(self):
        """Freeze the parameters to make them untrainable.
        """
        pass
    
    