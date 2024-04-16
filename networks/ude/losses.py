import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def calc_cond_embedding_similarity_loss_intra_domain(
    src_emb: torch.Tensor, 
    trg_emb: torch.Tensor, 
    normalize: bool = False
):
    """Calculate the intra-domain embedding similarity. 
    :param src_emb: [B, D]
    :param trg_emb: [B, D]
    """
    if normalize:
        norm_src_emb = F.normalize(src_emb, dim=-1)
        norm_trg_emb = F.normalize(trg_emb, dim=-1)
        src_similarity = F.cosine_similarity(norm_src_emb[:, None], 
                                             norm_src_emb[None], dim=-1)
        trg_similarity = F.cosine_similarity(norm_trg_emb[:, None], 
                                             norm_trg_emb[None], dim=-1)
    else:
        src_similarity = F.cosine_similarity(src_emb[:, None], 
                                             src_emb[None], dim=-1)
        trg_similarity = F.cosine_similarity(trg_emb[:, None], 
                                             trg_emb[None], dim=-1)
    
    # Calculate the loss between two similarity matrixs
    loss = F.l1_loss(src_similarity, trg_similarity, reduction="mean")
    return loss

def calc_cond_embedding_similarity_loss_inter_domain(
    src_emb: torch.Tensor, 
    trg_emb: torch.Tensor, 
    normalize: bool = False
):
    """Calculate the inter-domain embedding similarity. 
    :param src_emb: [B, D]
    :param trg_emb: [B, D]
    """
    batch_size = src_emb.size(0)
    mask = torch.eye(batch_size, batch_size).float().to(src_emb.device)
    
    if normalize:
        norm_src_emb = F.normalize(src_emb, dim=-1)
        norm_trg_emb = F.normalize(trg_emb, dim=-1)
        similarity = F.cosine_similarity(norm_src_emb[:, None], 
                                         norm_trg_emb[None], dim=-1)
    else:
        similarity = F.cosine_similarity(src_emb[:, None], 
                                         trg_emb[None], dim=-1)
    
    similarity = similarity * (1. - mask)
    loss = similarity.mean()
    return loss

def calc_cond_embedding_similarity_loss_reverse_augmentation(
    inp_emb: torch.Tensor, 
    normalize: bool = False
):
    """Calculate the similarity loss between embeddings encoded 
    from description with and without suffix prompts describing 'reverse'.
    :param inp_emb: [B, D]
    """
    batch_size = inp_emb.size(0)
    emb_a = inp_emb[:batch_size//2]
    emb_b = inp_emb[batch_size//2:]
    similarity = F.cosine_similarity(emb_a, emb_b, dim=-1)
    loss = similarity.mean()
    return loss

def calc_cond_motion_embedding_similarity_loss_reverse_augmentation(
    cond_emb: torch.Tensor, 
    motion: torch.Tensor, 
    motion_model: nn.Module, 
    normalize: bool = False
):
    """Calculate the similarity between motion embedding and condition embedding.
    :param cond_emb: [B, D]
    :motion: [B, T, C]
    :motion_model: the motion encoder model
    """
    with torch.no_grad():
        motion = F.interpolate(motion.detach().permute(0, 2, 1), 
                               size=160, align_corners=True, mode="linear")
        motion = motion.permute(0, 2, 1)
        mot_emb = motion_model(motion).loc[:, 0]
        if normalize:
            mot_emb = F.normalize(mot_emb, dim=-1).detach()
    
    batch_size = mot_emb.size(0)
    mot_similarity = F.cosine_similarity(mot_emb[:batch_size//2], mot_emb[batch_size//2:], dim=-1)
    if normalize:
        cond_emb = F.normalize(cond_emb, dim=-1)
    cond_similarity = F.cosine_similarity(cond_emb[:batch_size//2], cond_emb[batch_size//2:], dim=-1)
    loss = F.l1_loss(cond_similarity, mot_similarity, reduction="mean")
    return loss

def calc_cross_domain_cond_motion_embedding_similarity_loss_reverse_augmentation(
    cond_emb: torch.Tensor, 
    motion: torch.Tensor, 
    motion_model: nn.Module, 
    normalize: bool = False
):
    """Calculate the similarity between motion embedding and condition embedding.
    :param cond_emb: [B, D]
    :motion: [B, T, C]
    :motion_model: the motion encoder model
    """
    with torch.no_grad():
        motion = F.interpolate(motion.detach().permute(0, 2, 1), 
                               size=160, align_corners=True, mode="linear")
        motion = motion.permute(0, 2, 1)
        mot_emb = motion_model(motion).loc[:, 0]
        if normalize:
            mot_emb = F.normalize(mot_emb, dim=-1).detach()
    
    if normalize:
        cond_emb = F.normalize(cond_emb, dim=-1)
    similarity = F.cosine_similarity(cond_emb, mot_emb, dim=-1)
    loss = 1. - similarity.mean()
    return loss

def calc_cond_magnitude(
    cond_emb: torch.Tensor, 
    tok_emb: torch.Tensor, 
    normalize=False
):
    pass

def calc_z_regularization(z_emb, tok_emb, threshold=0.1):
    """Calculate the z embedding regularization.
    We want the norm of z_emb to be around threshold of tok_emb's norm: 
     - norm(z_emb) / norm(tok_emb) ~= threshold
    """
    loss = 0.0
    tok_emb_norm = torch.norm(tok_emb, dim=-1)  # [N]
    z_is_none = False
    for key, val in z_emb.items():
        if val is None: 
            z_is_none = True
            continue
        val_norm = torch.norm(val, dim=-1)
        loss = loss + val_norm.mean() / (tok_emb_norm.mean() + 1e-07)
    
    loss /= len(z_emb)
    if z_is_none:
        return loss
    else:
        return torch.abs(loss - threshold)
        
LOSS_MAPPING = {
    "cond_intra": partial(calc_cond_embedding_similarity_loss_intra_domain), 
    "cond_inter": partial(calc_cond_embedding_similarity_loss_inter_domain), 
    "cond_cross": partial(calc_cross_domain_cond_motion_embedding_similarity_loss_reverse_augmentation), 
    "cond_reverse_cross": partial(calc_cond_motion_embedding_similarity_loss_reverse_augmentation), 
    "cond_z_reg": partial(calc_z_regularization), 
}   

def get_losses(name):
    try:
        return LOSS_MAPPING[name]
    except:
        pass
     
    
    