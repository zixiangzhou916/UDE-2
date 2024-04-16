import os, sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import Tuple, Union, List, Any
from pkg_resources import packaging
from transformers import (
    AutoTokenizer, 
    CLIPTextModel
)
from networks.ude.simple_tokenizer import SimpleTokenizer as _Tokenizer 

__all__ = ["available_models", "load", "tokenize"]
# bpe_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 
#                       "pretrained-model", "openai", "clip", "tokenizer", "bpe_simple_vocab_16e6.txt.gz")
_tokenizer = _Tokenizer(bpe_path="pretrained_models/openai/clip-vit-base-patch32/bpe_simple_vocab_16e6.txt.gz")

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

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

    def encode(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x

class CLIPModel(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection       

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def extract_text_embedding(self, text):
        
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        return x

    def extract_image_embedding(self, image):
        x = self.visual.encode(image.type(self.dtype))
        return x

    def encode_text_from_onehot(self, text_prob):
        """
        :param text_prob: [batch_size, nframes, dim]
        """
        text = text_prob.max(-1)[1]
        x = torch.matmul(text_prob, self.token_embedding.weight)
        
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection       

        return x
   
class CLIP(nn.Module):
    def __init__(self, conf):
        super(CLIP, self).__init__()
        self.conf = conf
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        for key in conf.keys():
            if "cache_dir" not in key: continue
            if not os.path.exists(conf[key]): os.makedirs(conf[key])
        
        # print(conf["model_name"], conf["tokenizer_cache_dir"])
        print("Tokenizer building from pretrained...")
        # self.tokenizer = AutoTokenizer.from_pretrained(conf["model_name"], 
        #                                                cache_dir=conf["tokenizer_cache_dir"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=conf["tokenizer_cache_dir"])
        # self.model = CLIPTextModel.from_pretrained(conf["model_name"], 
        #                                            cache_dir=conf["model_cache_dir"])
        self.model = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path=conf["model_cache_dir"])

        self.model.eval()
        self.model = self.model.to(self.device)
        
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
        with torch.no_grad():
            if "padding_len" in self.conf.keys():
                padding_len = self.conf["padding_len"]
            else:
                padding_len = 77
            if "mask_padded" in self.conf.keys():
                mask_padded = self.conf["mask_padded"]
            else:
                mask_padded = False
                
            hidden_states = []
            masks = []
            lengths = []
            inputs = self.tokenizer(all_sens, padding=True, return_tensors="pt", 
                                    truncation=True)
            inputs = {key: val[:, :padding_len].to(self.device) for key, val in inputs.items()}
            # Pad the inputs so that we have fixed lengths
            t = inputs["input_ids"].size(1)
            if t < padding_len:
                ids_pad = torch.zeros(len(all_sens), padding_len-t).long().to(self.device)
                mask_pad = torch.zeros(len(all_sens), padding_len-t).long().to(self.device)
                inputs["input_ids"] = torch.cat([inputs["input_ids"], ids_pad], dim=1)
                inputs["attention_mask"] = torch.cat([inputs["attention_mask"], ids_pad], dim=1)
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            pooler_output = outputs.pooler_output
            
            if mask_padded:
                last_hidden_state *= inputs["attention_mask"].float().unsqueeze(-1)
        
        if not return_pooler:
            return last_hidden_state, inputs["attention_mask"].unsqueeze(dim=1).bool()
        else:
            return last_hidden_state, pooler_output, inputs["attention_mask"].unsqueeze(dim=1).bool()
    
class CLIPv2(nn.Module):
    def __init__(self, conf):
        super(CLIPv2, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.embed_dim = 512
        self.image_resolution = 224
        self.vision_layers = 12
        self.vision_width = 768
        self.vision_patch_size = 32
        self.context_length = 77
        self.vocab_size = 49408
        self.transformer_width = 512
        self.transformer_heads = 8
        self.transformer_layers = 12
        
        self.model = CLIPModel(
            self.embed_dim,
            self.image_resolution, self.vision_layers, 
            self.vision_width, self.vision_patch_size,
            self.context_length, self.vocab_size, 
            self.transformer_width, self.transformer_heads, 
            self.transformer_layers
        )
        # convert_weights(self.model)
        checkpoint = torch.jit.load(conf["model_cache_dir"], map_location=torch.device("cpu")).eval()
        state_dict = checkpoint.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in state_dict:
                del state_dict[key]
        self.model.load_state_dict(state_dict, strict=True)
        
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
        with torch.no_grad():
            text_tokens = tokenize(all_sens, truncate=True).to(self.device)
            attn_mask = get_clip_textencoder_mask(text_tokens)   # [batch_size, 1, num_frames]
            last_hidden_state = self.model.extract_text_embedding(text_tokens)  # [batch_size, num_frames, num_dims]
            
        if not return_pooler:
            # mask out the padding parts
            last_hidden_state = last_hidden_state.float()
            last_hidden_state *= attn_mask.permute(0, 2, 1).float()
            return last_hidden_state, attn_mask
        else:
            pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0]), text_tokens.argmax(dim=-1)] @ self.model.text_projection  
            last_hidden_state = last_hidden_state.float()     
            last_hidden_state *= attn_mask.permute(0, 2, 1).float()
            return last_hidden_state, pooled_output.float(), attn_mask
         
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.getcwd())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    conf = {
        "model_cache_dir": "networks/ude_v2/pretrained-model/openai/clip/model/ViT-B-32.pt"
    }
    # conf = {
    #     "model_name": "openai/clip-vit-base-patch32", 
    #     "conf_cache_dir": "networks/ude_v2/pretrained-model/openai/clip-vit-base-patch32/conf/", 
    #     "tokenizer_cache_dir": "networks/ude_v2/pretrained-model/openai/clip-vit-base-patch32/tokenizer/", 
    #     "model_cache_dir": "networks/ude_v2/pretrained-model/openai/clip-vit-base-patch32/model/", 
    #     "print": False, 
    #     "padding_len": 25, 
    #     "mask_padded": True
    # }
    
    model = CLIPv2(conf=conf)
    model.to(device)
    model.eval()
    
    for name, param in model.state_dict().items():
        print(name,"|", param.min().item(), "|", param.max().item())
    
    texts = [
        "a person starts walking forward and then begins to run.",
        "a person throwing something at shoulder height.",
        "someone scrolls from right to left and then stands",
        "a person walks forward at moderate pace.",
        # "a person dances and then runs backwards and forwards.",
        # "a person grabbed soemthing and put it somewhere",
        # "person standing to the side and checking his or her watch.",
        # "character is moving slowly before moving into a moderate jog.",
        # "a person strides forward for a few steps, then slows and stops.",
        # "swinging arm down then standing still.",
        # "a person runs forward and jumps over something, then turns around and jumps back over it.",
        # "walking forward then stopping.",
        # "a person hops on their left foot then their right",
        # "a man is is doing basketball drills.",
        # "the person is doing a military crawl across the floor",
        # "a person runs forward and jumps over something, then turns around and jumps back over it. a person runs forward and jumps over something, then turns around and jumps back over it. a person runs forward and jumps over something, then turns around and jumps back over it, jumps back over it. a person runs forward and jumps over something, then turns around and jumps back over it, jumps back over it. a person runs forward and jumps over something, then turns around and jumps back over it.",
    ]
    
    refs = [
        # "a man starts walking forward, then he begins to run."
        "one person is crawlling."
    ]
    # model.score(refs, texts)
    embeds, masks = model(texts, mask=None)
    print(embeds.shape, "|", masks.shape)
    print(embeds)
    # print('-' * 10)
    # print(masks[0])