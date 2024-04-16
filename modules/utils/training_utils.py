import torch
import torch.nn as nn
import torch.nn.functional as F

def load_partial(model, checkpoint, logger=None):
        loaded_params = dict()
        for name, val in checkpoint.items():
            name_new = name.replace('module.', '') if 'module.' in name else name
            loaded_params[name_new] = val
                
        model_params = dict()
        num_condition_encoder = 0
        for name, val in model.state_dict().items():
            name_new = name.replace('module.', '') if 'module.' in name else name
            model_params[name_new] = val

        valid_params = dict()
        valid_num_condition_encoder = 0
        for src_name, src_val in loaded_params.items():
            if src_name not in model_params.keys():
                continue
            src_val_shape = ', '.join(map(str, src_val.size()))
            dst_val = model_params[src_name]
            dst_val_shape = ', '.join(map(str, dst_val.size()))
            if src_val_shape != dst_val_shape:
                print("shape of {:s} does not match: {:s} <-> {:s}".format(src_name, src_val_shape, dst_val_shape))
                continue
            suffix = 'module.' if hasattr(model, "module") else ''
            valid_params[suffix + src_name] = src_val
            
        # assert valid_num_condition_encoder == num_condition_encoder
        if logger is not None:
            logger.info(' --- loading {:.5f}% params'.format(len(valid_params) / len(model_params) * 100.))
        else:
            print(' --- loading {:.5f}% params'.format(len(valid_params) / len(model_params) * 100.))
        model.load_state_dict(valid_params, strict=False)
            
        return model
    
def load_partial_embedding(model, checkpoint, logger=None):
        """Partially load codebook embedding weights.
        """
        loaded_params = dict()
        for name, val in checkpoint.items():
            name_new = name.replace('module.', '') if 'module.' in name else name
            loaded_params[name_new] = val
                
        model_params = dict()
        num_condition_encoder = 0
        for name, val in model.state_dict().items():
            name_new = name.replace('module.', '') if 'module.' in name else name
            model_params[name_new] = val
            
        valid_params = dict()
        for src_name, src_val in loaded_params.items():
            if src_name not in model_params.keys():
                continue
            loaded_num_embedding = src_val.size(0)
            loaded_embedding_dim = src_val.size(1)
            model_num_embedding = model_params[src_name].size(0)
            model_embedding_dim = model_params[src_name].size(1)
            num_embedding = min(loaded_num_embedding, model_num_embedding)
            embedding_dim = min(loaded_embedding_dim, model_embedding_dim)
                
            embedding = model_params[src_name].clone()
            embedding[:num_embedding, :embedding_dim] = src_val[:num_embedding, :embedding_dim]
            suffix = 'module.' if hasattr(model, "module") else ''
            valid_params[suffix + src_name] = embedding
            
        # assert valid_num_condition_encoder == num_condition_encoder
        if logger is not None:
            logger.info(' --- loading {:.5f}% params'.format(len(valid_params) / len(model_params) * 100.))
        else:
            print(' --- loading {:.5f}% params'.format(len(valid_params) / len(model_params) * 100.))
        model.load_state_dict(valid_params, strict=False)
        
        return model
    
def convert_smpl2joints(
    smpl_model, body_pose, **kwargs
):
    """
    :param smplx_model: 
    :param body_pose: [batch_size, nframes, 75]
    """
    B, T = body_pose.shape[:2]
    device = body_pose.device
    
    transl = body_pose[..., :3]
    global_orient = body_pose[..., 3:6]
    body_pose = body_pose[..., 6:]
    
    output = smpl_model(
        global_orient=global_orient.reshape(B*T, 1, -1), 
        body_pose=body_pose.reshape(B*T, -1, 3), 
        transl=transl.reshape(B*T, -1)
    )
    
    joints = output.joints.reshape(B, T, -1, 3)
    vertices3d = output.vertices.reshape(B, T, -1, 3)
    
    return {"joints": joints[:, :, :24], "vertices": vertices3d}

def convert_smplx2joints(
    smplx_model, body_pose, left_pose, right_pose, **kwargs
):
    """
    :param smplx_model: 
    :param body_pose: [batch_size, nframes, 69]
    :param left_pose: [batch_size, nframes, 12]
    :param right_pose: [batch_size, nframes, 12]
    """
    B, T = body_pose.shape[:2]
    device = body_pose.device
    
    transl = body_pose[..., :3]
    global_orient = body_pose[..., 3:6]
    body_pose = body_pose[..., 6:]
    if body_pose.shape[-1] == 69:
        body_pose = body_pose[..., :-6]
    betas = torch.zeros(B*T, 300).float().to(device)
    expression = torch.zeros(B*T, 100).float().to(device)
    jaw_pose = torch.zeros(B*T, 3).float().to(device)
    leye_pose = torch.zeros(B*T, 3).float().to(device)
    reye_pose = torch.zeros(B*T, 3).float().to(device)
    
    joints, vertices3d = [], []
    for i in range(B*T):
        out = smplx_model(
            betas=betas[i:i+1], 
            global_orient=global_orient.reshape(B*T, 1, -1)[i:i+1], 
            body_pose=body_pose.reshape(B*T, -1)[i:i+1], 
            left_hand_pose=left_pose.reshape(B*T, -1)[i:i+1], 
            right_hand_pose=right_pose.reshape(B*T, -1)[i:i+1], 
            expression=expression[i:i+1], 
            jaw_pose=jaw_pose[i:i+1], 
            leye_pose=leye_pose[i:i+1], 
            reye_pose=reye_pose[i:i+1],
            transl=transl.reshape(B*T, -1)[i:i+1]
        )
        joints.append(out.joints.reshape(1, -1, 3))
        vertices3d.append(out.vertices.reshape(1, -1, 3))
    joints = torch.cat(joints, dim=0).reshape(B, T, -1, 3)
    vertices3d = torch.cat(vertices3d, dim=0).reshape(B, T, -1, 3)
    
    return {"joints": joints, "vertices": vertices3d}

def calc_semantic_enhancement_loss(
    cond_embed: torch.Tensor, 
    motion_embed: torch.Tensor, 
    normalize: bool = True
):
    """
    :param cond_embed: [batch_size, num_dim] or [batch_size, 1, num_dim]
    :param motion_embed: [batch_size, num_dim] or [batch_size, 1, num_dim]
    """
    def squeeze(x):
        if x.dim() == 3:
            return x.squeeze(dim=1)
        else:
            return x
    cond_embed = squeeze(cond_embed)
    motion_embed = squeeze(motion_embed)
    if normalize:
        cond_embed = F.normalize(cond_embed, dim=-1)
        motion_embed = F.normalize(motion_embed, dim=-1)
        
    # Embedding similarity betweeen motion and condition
    cross_similarity_1 = F.cosine_similarity(cond_embed, motion_embed, dim=-1)                    # [B]
    # # Pairwise embedding similarity of mini-batch
    # motion_similarity = F.cosine_similarity(motion_embed[:, None], motion_embed[None], dim=-1)  # [B, B]
    # cond_similarity = F.cosine_similarity(cond_embed[:, None], cond_embed[None], dim=-1)        # [B, B]
    # cross_similarity_2 = F.cosine_similarity(motion_similarity, cond_similarity, dim=-1)        # [B]
    
    # Calculate the losses
    loss1 = (1. - cross_similarity_1).mean()
    # loss2 = (1. - cross_similarity_2).mean()
    return {"sem_enh_1": loss1}
