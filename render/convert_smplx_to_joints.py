import os, sys, argparse
sys.path.append(os.getcwd())
from tqdm import tqdm
import json
import numpy as np
import torch
import torch.nn.functional as F
from networks import smplx_code

smplx_cfg = dict(
    model_path="pretrained_models/human_models/smpl-x/SMPLX_NEUTRAL_2020.npz", 
    model_type="smplx", 
    gender="neutral", 
    use_face_contour=True, 
    use_pca=True, 
    flat_hand_mean=False, 
    use_hands=True, 
    use_face=True, 
    num_pca_comps=12, 
    num_betas=300, 
    num_expression_coeffs=100,
)

smpl_cfg = dict(
    model_path="pretrained_models/human_models/smpl/SMPL_NEUTRAL.pkl", 
    model_type="smpl", 
    gender="neutral", 
    batch_size=1,
)

mano_left_cfg = dict(
    model_path="pretrained_models/human_models/mano/models/MANO_LEFT.pkl", 
    model_type="mano", 
    is_rhand=False, 
    gender="neutral", 
    num_pca_comps=12
)

mano_right_cfg = dict(
    model_path="pretrained_models/human_models/mano/models/MANO_RIGHT.pkl", 
    model_type="mano", 
    is_rhand=True, 
    gender="neutral", 
    num_pca_comps=12
)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
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
    
    return {"joints": joints[:, :, :24], "vertices": vertices3d, "faces": smpl_model.faces}

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
    elif body_pose.shape[-1] == 75:
        body_pose = body_pose[..., :-12]
    
    betas = torch.zeros(B*T, 300).float().to(device)
    expression = torch.zeros(B*T, 100).float().to(device)
    jaw_pose = torch.zeros(B*T, 3).float().to(device)
    leye_pose = torch.zeros(B*T, 3).float().to(device)
    reye_pose = torch.zeros(B*T, 3).float().to(device)
    
    output = smplx_model(
        betas=betas, 
        global_orient=global_orient.reshape(B*T, 1, -1), 
        body_pose=body_pose.reshape(B*T, -1), 
        left_hand_pose=left_pose.reshape(B*T, -1), 
        right_hand_pose=right_pose.reshape(B*T, -1), 
        expression=expression, 
        jaw_pose=jaw_pose, 
        leye_pose=leye_pose, 
        reye_pose=reye_pose,
        transl=transl.reshape(B*T, -1)
    )

    joints = output.joints.reshape(B, T, -1, 3)
    vertices3d = output.vertices.reshape(B, T, -1, 3)
    
    return {"joints": joints, "vertices": vertices3d, "faces": smplx_model.faces}

def convert_mano2joints(
    mano_model, hand_pose, **kwargs
):
    B, T = hand_pose.shape[:2]
    device = hand_pose.device
    
    global_orient = torch.zeros(B*T, 3).float().to(hand_pose.device)
    transl = torch.zeros(B*T, 3).float().to(hand_pose.device)
    betas = torch.zeros(B*T, 10).float().to(hand_pose.device)
    hand_pose = hand_pose.reshape(B*T, -1)
    
    output = mano_model(
        betas=betas, 
        global_orient=global_orient, 
        transl=transl, 
        hand_pose=hand_pose)
    
    joints = output.joints.reshape(B, T, -1, 3)
    vertices3d = output.vertices.reshape(B, T, -1, 3)
    
    return {"joints": joints[:, :, :24], "vertices": vertices3d, "faces": mano_model.faces}

def main(input_dir, output_dir, targ_tids, convert_gt=False, step_size=1):
    # Make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Build SMPL and SMPLX models
    smpl_model = smplx_code.create(**smpl_cfg)
    smplx_model = smplx_code.create(**smplx_cfg)
    mano_left_model = smplx_code.create(**mano_left_cfg)    
    mano_right_model = smplx_code.create(**mano_right_cfg)    
    
    output_file = []
    
    # Get all files
    all_files = [f for f in os.listdir(input_dir) if ".npy" in f]
    # Re-organize the files according to the TID
    files_reorg = {key: [] for key in targ_tids}
    for file in all_files:
        try:
            tid = file.split(".")[0].split("_")[1]
        except:
            tid = targ_tids[0]
        if tid not in targ_tids: 
            continue
        files_reorg[tid].append(file)
    
    for key, val in files_reorg.items():
        files_reorg[key] = sorted(val)
        
    for tid, files in files_reorg.items():
        for file in tqdm(files[::step_size], desc="SMPLX-to-MESH: TID = {:s}".format(tid)):
            cur_output_dir = os.path.join(output_dir, tid)
            if not os.path.exists(cur_output_dir):
                os.makedirs(cur_output_dir)
            
            data = np.load(os.path.join(input_dir, file), allow_pickle=True).item()
            name = data["caption"][0]
            output_file.append({file: name})
            
            if os.path.exists(os.path.join(cur_output_dir, file)):
                continue
            
            try:
                if convert_gt:
                    poses = {key+"_pose": torch.from_numpy(val).permute(0, 2, 1).float() for key, val in data["gt"].items()}
                else:
                    poses = {key+"_pose": torch.from_numpy(val).permute(0, 2, 1).float() for key, val in data["pred"].items()}
            except:
                if convert_gt:
                    poses = {"body_pose": torch.from_numpy(data["gt"]).permute(0, 2, 1).float()}
                else:
                    poses = {"body_pose": torch.from_numpy(data["motion"]).permute(0, 2, 1).float()}
                    
            if len(poses) == 3:
                result = convert_smplx2joints(smplx_model=smplx_model, **poses)
                lh_result = convert_mano2joints(mano_model=mano_left_model, hand_pose=poses["left_pose"])
                rh_result = convert_mano2joints(mano_model=mano_right_model, hand_pose=poses["right_pose"])
                pred = {
                    "body": result["joints"][0].data.cpu().numpy(), 
                    "left": lh_result["joints"][0].data.cpu().numpy(), 
                    "right": rh_result["joints"][0].data.cpu().numpy()
                }
                # faces = {
                #     "body": result["faces"], "left": lh_result["faces"], "right": rh_result["faces"]
                # }
            elif len(poses) == 1:
                result = convert_smpl2joints(smpl_model=smpl_model, **poses)
                pred = {
                    "body": result["joints"][0].data.cpu().numpy()
                }
                # faces = {
                #     "body": result["faces"]
                # }

            output = {
                "pred": pred, 
                "caption": data["caption"], 
                "color_labels": data.get("color_labels", None),
            }
            np.save(os.path.join(cur_output_dir, file), output)
                
    with open(os.path.join(output_dir, "filename_to_captions.json"), "w") as f:
        json.dump(output_file, f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='logs/ude/eval/exp3/output/t2m', help='')
    parser.add_argument('--output_dir', type=str, default='logs/ude/eval/exp3/joints/t2m', help='')
    parser.add_argument('--targ_tids', type=str, default='T0000,T0001,T0002', help='')
    parser.add_argument('--convert_gt', type=str2bool, default=False, help='')
    parser.add_argument('--step_size', type=int, default=1, help='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    targ_tids = args.targ_tids
    if targ_tids is not None:
        targ_tids = targ_tids.split(",")
    main(args.input_dir, args.output_dir, targ_tids, args.convert_gt, args.step_size)
