import os
import numpy as np
import torch
from networks.roma.utils import rotvec_slerp

def save_results(result, output_dir, batch_id, generation_id):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, "B{:04d}_T{:04d}.npy".format(batch_id, generation_id)), result)
    
def merge_motion_segments(segments, seg_len):
    merged = []
    for i in range(1, len(segments), 1):
        clip_1 = segments[i-1][0]  # [T, D]
        clip_2 = segments[i][0]    # [T, D]
        
        len_1 = clip_1.size(0)
        len_2 = clip_2.size(0)
        
        init_transl = clip_2[:1, :3]
        last_transl = clip_1[seg_len:seg_len+1, :3]
        offset = init_transl - last_transl
        clip_2[:, :3] -= offset
        
        num_rotvec = (clip_1.size(1) - 3) // 3
        num_steps = len_1 - seg_len
        interp_rotvecs = []
        interp_transl = []
        for j in range(0, num_steps, 1):
            steps = torch.Tensor([j / num_steps]).float().to(clip_1.device)
            # Slerp the rotvecs
            rotvecs_interp = rotvec_slerp(
                rotvec0=clip_1[j+seg_len, 3:].view(-1, 3), 
                rotvec1=clip_2[j, 3:].view(-1, 3), steps=steps)[0]
            interp_rotvecs.append(rotvecs_interp)
            # Lerp the transl
            transl_interp = torch.lerp(clip_1[j+seg_len:j+seg_len+1, :3], clip_2[j:j+1, :3], weight=steps)
            interp_transl.append(transl_interp) 
        
        interp_rotvecs = torch.stack(interp_rotvecs, dim=0).view(num_steps, -1)
        interp_transl = torch.cat(interp_transl, dim=0)  
        
        if len(merged) == 0:
            merged.append(clip_1[:seg_len])
        else:
            merged.append(clip_1[num_steps:seg_len])
        merged.append(torch.cat([interp_transl, interp_rotvecs], dim=-1))
        
    merged.append(clip_2[seg_len:])
    merged = torch.cat(merged, dim=0).unsqueeze(dim=0)
    return merged
    