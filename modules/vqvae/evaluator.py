import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import importlib
from funcs.logger import setup_logger
from funcs.comm_utils import get_rank
from datetime import datetime
from tqdm import tqdm
import yaml
import json

from modules.utils.training_utils import *
from networks import smplx_code

smplx_cfg = dict(
    model_path="networks/smpl-x/SMPLX_NEUTRAL_2020.npz", 
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
    model_path="networks/smpl/SMPL_NEUTRAL.pkl", 
    model_type="smpl", 
    gender="neutral", 
    batch_size=1,
)

""" VQ-VAE Evaluator """
class VQTokenizerEvaluator(object):
    def __init__(self, args, opt):
        self.args = args
        self.opt = opt
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.output_dir = os.path.join(self.args.eval_folder, self.args.eval_name, "output")
        if not os.path.exists(self.output_dir): 
            os.makedirs(self.output_dir)
        self.logger = setup_logger('UDE-2', self.output_dir, get_rank(), filename='vqvae_eval_log.txt')
        self.setup_models()
        self.setup_human_prior_model()
        self.setup_loaders()
        
    def setup_models(self):
        
        def build_model(model_conf, key, opt_conf, checkpoint):
            """Build VQVAE model."""
            encoder = importlib.import_module(model_conf["vq_encoder"]["arch_path"], package="networks").__getattribute__(
                model_conf["vq_encoder"]["arch_name"])(**model_conf["vq_encoder"])
            decoder = importlib.import_module(model_conf["vq_decoder"]["arch_path"], package="networks").__getattribute__(
                model_conf["vq_decoder"]["arch_name"])(**model_conf["vq_decoder"])
            quantizer = importlib.import_module(model_conf["quantizer"]["arch_path"], package="networks").__getattribute__(
                model_conf["quantizer"]["arch_name"])(**model_conf["quantizer"])
            """Load weights"""
            encoder.load_state_dict(checkpoint[key+"_vqencoder"], strict=True)
            self.logger.info("VQ-Encoder weights loaded")
            decoder.load_state_dict(checkpoint[key+"_vqdecoder"], strict=True)
            self.logger.info("VQ-Decoder weights loaded")
            quantizer.load_state_dict(checkpoint[key+"_quantizer"], strict=True)
            self.logger.info("Quantizer weights loaded")
            return encoder.to(self.device), decoder.to(self.device), quantizer.to(self.device)
        
        self.models = {}
        for key, model_conf in self.opt["model"].items():
            self.logger.info("Pretrained VQ-VAE weights are stored at: {:s}".format(self.opt["eval"]["checkpoints"][key]))
            checkpoint = torch.load(self.opt["eval"]["checkpoints"][key], map_location=torch.device("cpu"))
            self.models[key+"_vqencoder"], self.models[key+"_vqdecoder"], self.models[key+"_quantizer"] = \
                build_model(model_conf=model_conf, key=key, opt_conf=self.opt["train"], checkpoint=checkpoint)
                
    def setup_loaders(self):
        self.eval_loader, _ = importlib.import_module(
            ".ude.dataloader", package="dataloader").__getattribute__(
                "get_dataloader")(self.opt["data"]["dataset"], 
                                        self.opt["data"]["loader"]["test"], 
                                        meta_dir=None)
    
    def setup_human_prior_model(self):
        self.smplx_model = smplx_code.create(**smplx_cfg)
        self.smplx_model = self.smplx_model.to(self.device)
        
        self.smpl_model = smplx_code.create(**smpl_cfg)
        self.smpl_model = self.smpl_model.to(self.device)
        
    def add_translation(self, init_trans, trans, poses):
        """Add global translation to poses.
        """
        glob_trans = [init_trans[:, 0]]
        for i in range(1, trans.size(1), 1):
            glob_trans.append(
                glob_trans[-1] + trans[:, i]
            )
        glob_trans = torch.stack(glob_trans, dim=1)
        poses[:, :, :3] = glob_trans
        return poses
    
    def apply_initial_translation_and_orientation(self, motion, init_trans=None, init_orien=None):
        """Apply initial translation(if applicable) and 
        init_orientation(if applicable) to generated motion sequences.
        :param motion: [batch_size, seq_len, dim]
        :param init_trans: (optional) [batch_size, 1, 3]
        :param init_orien: (optional) [batch_size, 1, 3]
        """
        if motion.size(-1) == 12:
            # We dont conduct this process on hand motion sequence
            return motion
        if motion.size(-1) == 263:
            # We dont conduct this process for motion with 263 dimension
            return motion
        if init_trans is None and init_orien is None:
            # Both init_trans and init_orien are not applicable
            return motion
        elif init_trans is not None and init_orien is None:
            # Only init_trans is applicable
            offset = motion[:, :1, :3] - init_trans
            motion[:, :, :3] -= offset
            return motion
        elif init_trans is None and init_orien is not None:
            # Only init_orien is applicable
            pass
        elif init_trans is not None and init_orien is not None:
            # Both init_trans and init_orien are applicable
            pass
        else:
            raise ValueError
    
    def eval(self):
        for key, _ in self.models.items():
            self.models[key].eval()
        
        rec_losses = {}
        emb_losses = {}
        pos_losses = [] # joints position error
        vel_losses = [] # joints velocity error
        acc_losses = [] # joints acceleration error
        token_occupancy = [0 for _ in range(self.opt["model"]["body"]["quantizer"]["n_e"])]
        total_id = 0
        for batch_id, batch in enumerate(tqdm(self.eval_loader, desc="Evaluating")):
            if batch_id % 100 != 0: 
                continue
            gt_poses = {}
            rc_poses = {}
            for key, x in batch.items():
                if key+"_vqencoder" not in self.models.keys():
                    continue
                x = x.float().to(self.device)
                if self.opt["eval"]["strategy"] == "two_stage":
                    x_raw = x.clone()       # Make a copy of the raw input
                    batch_size, seq_len = x.shape[:2]
                    trans = x[..., :3].clone()
                    pose = x[..., 3:].clone()
                    offsets = trans[:, 1:] - trans[:, :-1]
                    zero_trans = torch.zeros(batch_size, 1, 3).float().to(x.device)
                    offsets = torch.cat([zero_trans, offsets], dim=1)
                    
                    x = torch.cat([offsets, pose], dim=-1) # root set to offsets relative previous frame
                else:
                    pass
                
                with torch.no_grad():
                    latent = self.models[key+"_vqencoder"](x)
                    tokens = self.models[key+"_quantizer"].map2index(latent)
                    emb_loss, vq_latents, _, _ = self.models[key+"_quantizer"](latent)
                    y = self.models[key+"_vqdecoder"](vq_latents)
                
                # Count token occupancy
                if "body" in key:
                    tokens = tokens.data.cpu().numpy()
                    for tok in tokens:
                        token_occupancy[tok] += 1
                
                if self.opt["eval"]["strategy"] == "two_stage":
                    y = self.apply_initial_translation_and_orientation(
                        motion=y, init_orien=None, 
                        init_trans=x_raw[:, :1, :3].to(self.device))
                    x = x_raw
                else:
                    pass
                
                rec_loss = F.l1_loss(x, y, reduction="mean")
                if key in rec_losses.keys():
                    rec_losses[key].append(rec_loss.item())
                else:
                    rec_losses[key] = [rec_loss.item()]
                if key in emb_losses.keys():
                    emb_losses[key].append(emb_loss.item())
                else:
                    emb_losses[key] = [emb_loss.item()]
                    
                if self.opt["eval"]["strategy"] == "two_stage":
                    # Recover the target x
                    x = torch.cat([trans, pose], dim=-1)
                
                gt_poses[key] = x.permute(0, 2, 1).data.cpu().numpy()
                rc_poses[key] = y.permute(0, 2, 1).data.cpu().numpy()
            
            # Convert to joints to calculate joints positional error and velocity error
            p_poses = {key+"_pose": torch.from_numpy(val).permute(0, 2, 1).float().to(self.device) for key, val in rc_poses.items()}
            g_poses = {key+"_pose": torch.from_numpy(val).permute(0, 2, 1).float().to(self.device) for key, val in gt_poses.items()}
            
            p_joints = convert_smpl2joints(self.smpl_model, **p_poses)["joints"]
            g_joints = convert_smpl2joints(self.smpl_model, **g_poses)["joints"]
            # if len(p_poses) == 1:
            #     p_joints = convert_smpl2joints(self.smpl_model, **p_poses)["joints"]
            #     g_joints = convert_smpl2joints(self.smpl_model, **g_poses)["joints"]
            # elif len(p_poses) == 3:
            #     p_joints = convert_smplx2joints(self.smplx_model, **p_poses)["joints"]
            #     g_joints = convert_smplx2joints(self.smplx_model, **g_poses)["joints"]
            p_vel = p_joints[:, 1:] - p_joints[:, :-1]
            p_acc = p_vel[:, 1:] - p_vel[:, :-1]
            g_vel = g_joints[:, 1:] - g_joints[:, :-1]
            g_acc = g_vel[:, 1:] - g_vel[:, :-1]
            pos_loss = F.l1_loss(p_joints, g_joints, reduction="mean")
            vel_loss = F.l1_loss(p_vel, g_vel, reduction="mean")
            acc_loss = F.l1_loss(p_acc, g_acc, reduction="mean")
            pos_losses.append(pos_loss.item())
            vel_losses.append(vel_loss.item())
            acc_losses.append(acc_loss.item())
            
            for j in range(x.shape[0]):
                output = {
                    "gt": {key: val[j:j+1] for key, val in gt_poses.items()}, 
                    "pred": {key: val[j:j+1] for key, val in rc_poses.items()}, 
                    "caption": [str(total_id)]
                }
                output_path = os.path.join(self.output_dir, "t2m")
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                np.save(os.path.join(output_path, "B{:004}.npy".format(total_id)), output)
                total_id += 1
            
            # output = {
            #     "gt": gt_poses, "pred": rc_poses, "caption": [str(batch_id)]
            # }
            # output_path = os.path.join(self.output_dir, "t2m")
            # if not os.path.exists(output_path):
            #     os.makedirs(output_path)
            # np.save(os.path.join(output_path, "B{:004}.npy".format(batch_id)), output)
                
        for key, loss in rec_losses.items():
            print("rec_loss: ", key, np.mean(loss))
        for key, loss in emb_losses.items():
            print("emb_loss: ", key, np.mean(loss))
        print('pos_loss: ', np.mean(pos_losses))
        print('vel_loss: ', np.mean(vel_losses))
        print('acc_loss: ', np.mean(acc_losses))
    
        with open(os.path.join(self.output_dir, "codebook_occupancy_body.json"), "w") as f:
            json.dump(token_occupancy, f)
        
        