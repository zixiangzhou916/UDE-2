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
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    print("Unable to import tensorboard")

from modules.utils.training_utils import *
from networks import smplx_code
from render.vqvae.render_skeleton import render_animation

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

""" VQ-VAE Trainer """
class VQTokenizerTrainer(object):
    def __init__(self, args, opt):
        self.args = args
        self.opt = opt
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.timestep = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        self.training_folder = os.path.join(self.args.training_folder, self.args.training_name, self.timestep)
        if not os.path.exists(self.training_folder): 
            os.makedirs(self.training_folder)
        self.logger = setup_logger('UDE-2', self.training_folder, get_rank(), filename='vqvae_log.txt')
        # self.writer = SummaryWriter(log_dir=self.training_folder)
        with open(os.path.join(self.training_folder, "config_vqvae.yaml"), 'w') as outfile:
            yaml.dump(self.opt, outfile, default_flow_style=False)
        
        self.epoch = 0
        self.global_step = 0
        self.dual_stream = False     # whether to train dual stream decoder
        self.start_dual = False 
        self.training_strategy = self.opt["train"].get("strategy", "two_stage") # 1. vanilla, 2. two_stage
        
        self.setup_models()
        self.load_checkpoints()
        # self.print_model()
        self.setup_loaders()
        self.setup_human_prior_model()
        
    def setup_models(self):
        
        def build_model(model_conf, key, opt_conf):
            """Build VQVAE model."""
            encoder = importlib.import_module(model_conf["vq_encoder"]["arch_path"], package="networks").__getattribute__(
                model_conf["vq_encoder"]["arch_name"])(**model_conf["vq_encoder"])
            decoder = importlib.import_module(model_conf["vq_decoder"]["arch_path"], package="networks").__getattribute__(
                model_conf["vq_decoder"]["arch_name"])(**model_conf["vq_decoder"])
            quantizer = importlib.import_module(model_conf["quantizer"]["arch_path"], package="networks").__getattribute__(
                model_conf["quantizer"]["arch_name"])(**model_conf["quantizer"])
            opt = optim.Adam(
                list(encoder.parameters()) + list(decoder.parameters()) + list(quantizer.parameters()), 
                # decoder.parameters(), # We only update decoder parameters
                lr=opt_conf["lr"], 
                weight_decay=self.opt["train"]["weight_decay"])
            scheduler = optim.lr_scheduler.StepLR(opt, step_size=opt_conf["step_lr"], gamma=opt_conf["gamma"])
            return encoder.to(self.device), decoder.to(self.device), quantizer.to(self.device), opt, scheduler
        
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        for key, model_conf in self.opt["model"].items():
            self.models[key+"_vqencoder"], self.models[key+"_vqdecoder"], self.models[key+"_quantizer"], \
                self.optimizers[key], self.schedulers[key] = \
                    build_model(model_conf=model_conf, key=key, opt_conf=self.opt["train"])
    
    def setup_human_prior_model(self):
        self.smplx_model = smplx_code.create(**smplx_cfg)
        self.smplx_model = self.smplx_model.to(self.device)
        
        self.smpl_model = smplx_code.create(**smpl_cfg)
        self.smpl_model = self.smpl_model.to(self.device)
        
    def setup_loaders(self):
        self.train_loader, _ = importlib.import_module(
            ".ude.dataloader", package="dataloader").__getattribute__(
                "get_dataloader")(self.opt["data"]["dataset"], 
                                        self.opt["data"]["loader"]["train"], 
                                        meta_dir=None)
        self.val_loader, _ = importlib.import_module(
            ".ude.dataloader", package="dataloader").__getattribute__(
                "get_dataloader")(self.opt["data"]["dataset"], 
                                        self.opt["data"]["loader"]["vald"], 
                                        meta_dir=None)
    
    def load_checkpoints(self):
        def load(key, ckpt_path):
            try:
                checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
                self.models[key+"_vqencoder"] = load_partial(self.models[key+"_vqencoder"], checkpoint[key+"_vqencoder"], logger=self.logger)
                self.models[key+"_vqdecoder"] = load_partial(self.models[key+"_vqdecoder"], checkpoint[key+"_vqdecoder"], logger=self.logger)
                self.models[key+"_quantizer"] = load_partial_embedding(self.models[key+"_quantizer"], checkpoint[key+"_quantizer"], logger=self.logger)
                self.logger.info("checkpoint of {:s} loading from {:s} successful!".format(key, ckpt_path))
            except:
                self.logger.info("checkpoint of {:s} loading from {:s} failed!".format(key, ckpt_path if ckpt_path is not None else "None"))
        
        for key, ckpt_path in self.opt["train"]["checkpoints"].items():
            load(key=key, ckpt_path=ckpt_path)
            
    def save_checkpoints(self, epoch, name):
        
        def save_model(key, epoch, global_step, name):
            state = {
                key+"_vqencoder": self.models[key+"_vqencoder"].state_dict(), 
                key+"_vqdecoder": self.models[key+"_vqdecoder"].state_dict(), 
                key+"_quantizer": self.models[key+"_quantizer"].state_dict(), 
                "opt": self.optimizers[key].state_dict(), 
                "epoch": epoch, 
                "global_step": global_step
            }
            save_dir = os.path.join(self.training_folder, "checkpoints")
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            torch.save(state, os.path.join(save_dir, name))
        
        for key in self.opt["train"]["part_to_train"]:
            save_model(key, epoch, self.global_step, "{:s}_{:s}".format(key, name))
    
    def reset_codebook(self):
        """Try to reset part of the code embeddings to improve the usage of codebook.
        """
        log_str = "=" * 50 + "start to re-initialize the weights" + "=" * 50
        self.logger.info(log_str)
        token_occupancy = {key: {} for key in ["body", "left", "right"]}
        for batch in tqdm(self.train_loader, desc="Resetting codebook"):
            for key, x in batch.items():
                if key+"_vqencoder" not in self.models.keys(): 
                    continue
                with torch.no_grad():
                    if self.opt["train"].get("strategy", "vanilla") == "vanilla":
                        pass
                    elif self.opt["train"].get("strategy", "vanilla") == "two_stage":
                        trans = x[..., :3].clone()  # Global translation
                        pose = x[..., 3:].clone()   # Rotation vectors
                        offsets = trans[:, 1:] - trans[:, :-1]
                        zero_trans = torch.zeros(x.size(0), 1, 3).float().to(x.device)
                        offsets = torch.cat([zero_trans, offsets], dim=1)
                        x = torch.cat([offsets, pose], dim=-1) # Set root to offsets relative previous frame
                        
                    latent = self.models[key+"_vqencoder"](x.detach().to(self.device).float())
                    token = self.models[key+"_quantizer"].map2index(latent)
                    token = token.data.cpu().numpy()
                    for k in token:
                        if k in token_occupancy[key].keys():
                            token_occupancy[key][k] += 1
                        else:
                            token_occupancy[key][k] = 1
        
        # Sort the occupancy
        for key, occupancy in token_occupancy.items():
            if len(occupancy) == 0: continue
            num_tokens = self.models[key+"_quantizer"].n_e
            total_occupancy = [0] * num_tokens
            for k, cnt in occupancy.items():
                total_occupancy[k] = cnt
            sort_index = np.argsort(np.array(total_occupancy))
            
            codebook_embedding = self.models[key+"_quantizer"].embedding.weight.clone()
            
            i = 0
            j = num_tokens - 1
            while total_occupancy[sort_index[i]] == 0:
                noise = torch.rand_like(self.models[key+"_quantizer"].embedding.weight[0]).uniform_(-0.00001, 0.00001)
                codebook_embedding[sort_index[i]] = self.models[key+"_quantizer"].embedding.weight[sort_index[j]] + noise
                i += 1
                j -= 1
            
            # Assign new codebook embedding weight
            self.models[key+"_quantizer"].embedding.weight.data = codebook_embedding
            
        log_str = "=" * 50 + "weights re-initialization done" + "=" * 50
        self.logger.info(log_str)
    
    @staticmethod
    def calc_skeleton_loss(gt, pred):
        pred_vel = pred[:, 1:] - pred[:, :-1]
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        gt_vel = gt[:, 1:] - gt[:, :-1]
        gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
        pos_loss = F.mse_loss(pred, gt, reduction="mean")
        vel_loss = F.mse_loss(pred_vel, gt_vel, reduction="mean")
        acc_loss = F.mse_loss(pred_acc, gt_acc, reduction="mean")
        return pos_loss, vel_loss, acc_loss
    
    def run_one_step_one_stage(self, x, key="left"):
        # Run the VQ-VAE
        latent = self.models[key+"_vqencoder"](x)
        embedding_loss, vq_latents, emb_indices, perplexity = self.models[key+"_quantizer"](latent)
        glob_output = self.models[key+"_vqdecoder"](vq_latents)                     # Global poses
        
        # Reconstruction loss (smpl/smplx)
        global_rec_loss = F.l1_loss(glob_output, x, reduction="mean")

        
        recon_losses_info = {
            key+"_g_rc": global_rec_loss
        }
        losses_info = {
            key+"_embed": embedding_loss, 
            key+"_perplexity": perplexity
        }
        losses_info.update(recon_losses_info)
        
        total_recon_loss = 0.0
        for _, val in recon_losses_info.items():
            total_recon_loss += val
        total_loss = total_recon_loss + embedding_loss
        
        return glob_output, total_loss, losses_info
        
    def run_one_step_two_stage(self, x, key="body"):
        """We set the root as the shift relative to previous frame, and the vq-decoder will 
        directly decode the codes back to global motion sequence.
        root of every pose to (0, 0, 0).
        """
        batch_size, seq_len = x.shape[:2]
        trans = x[..., :3].clone()  # Global translation
        pose = x[..., 3:].clone()   # Rotation vectors
        
        offsets = trans[:, 1:] - trans[:, :-1]
        zero_trans = torch.zeros(batch_size, 1, 3).float().to(x.device)
        offsets = torch.cat([zero_trans, offsets], dim=1)
        
        inputs = torch.cat([offsets, pose], dim=-1) # Set root to offsets relative previous frame
        
        # Run the VQ-VAE
        latent = self.models[key+"_vqencoder"](inputs)
        embedding_loss, vq_latents, emb_indices, perplexity = self.models[key+"_quantizer"](latent)
        glob_output = self.models[key+"_vqdecoder"](vq_latents)                     # Global poses
        local_output = self.models[key+"_vqdecoder"].get_sublocal_outputs()         # Sub-local poses
        sub_glob_output = self.models[key+"_vqdecoder"].get_subglobal_outputs()     # Sub-global poses
        
        # Put the initial frame of both output and input at the same location, so we can calculate the L1 loss fairly
        offset_glob = glob_output[:, :1, :3] - x[:, :1, :3]             # [B, 1, 3]
        offset_sub_glob = sub_glob_output[:, :1, :3] - x[:, :1, :3]     # [B, 1, 3]

        glob_output[:, :, :3] = glob_output[:, :, :3] - offset_glob
        sub_glob_output[:, :, :3] = sub_glob_output[:, :, :3] - offset_sub_glob
        
        # Reconstruction loss (smpl/smplx)
        local_rec_loss = F.l1_loss(inputs, local_output, reduction="mean")
        global_rec_loss = F.l1_loss(x, glob_output, reduction="mean")
        sub_global_rec_loss = F.l1_loss(x, sub_glob_output, reduction="mean")
        # # Reconstruction loss (joints)
        # gt_results =  convert_smpl2joints(self.smpl_model, body_pose=x)
        # glob_results = convert_smpl2joints(self.smpl_model, body_pose=glob_output)
        # sub_glob_results = convert_smpl2joints(self.smpl_model, body_pose=sub_glob_output)
        
        # glob_pos_loss, glob_vel_loss, glob_acc_loss = self.calc_skeleton_loss(gt_results["joints"], glob_results["joints"])
        # sub_glob_pos_loss, sub_glob_vel_loss, sub_glob_acc_loss = self.calc_skeleton_loss(gt_results["joints"], sub_glob_results["joints"])
        
        recon_losses_info = {
            key+"_l_rc": local_rec_loss, 
            key+"_g_rc": global_rec_loss, 
            key+"_sg_rc": sub_global_rec_loss, 
            # key+"_g_pos_rc": glob_pos_loss, 
            # key+"_g_vel_rc": glob_vel_loss, 
            # key+"_g_acc_rc": glob_acc_loss, 
            # key+"_sg_pos_rc": sub_glob_pos_loss, 
            # key+"_sg_vel_rc": sub_glob_vel_loss, 
            # key+"_sg_acc_rc": sub_glob_acc_loss, 
        }
        losses_info = {
            key+"_embed": embedding_loss, 
            key+"_perplexity": perplexity
        }
        losses_info.update(recon_losses_info)
        
        total_recon_loss = 0.0
        for _, val in recon_losses_info.items():
            total_recon_loss += val
        total_loss = total_recon_loss + embedding_loss
        
        return glob_output, total_loss, losses_info
    
    def train_one_step(self, batch, epoch, iter, total_iters):
        # if "dual_stream" in self.opt["train"].keys():
        #     self.dual_stream = True
            
        for key, x in batch.items():
            if key not in self.opt["train"]["part_to_train"]: continue
            # Train
            self.optimizers[key].zero_grad()
            if key == "body" and self.opt["train"]["strategy"] == "two_stage":
                y, total_loss, losses_info = self.run_one_step_two_stage(x.detach().to(self.device).float(), key=key)
            else:
                y, total_loss, losses_info = self.run_one_step_one_stage(x.detach().to(self.device).float(), key=key)
            
            # clip_grad_norm_(self.models[key+"_vqencoder"].parameters(), 0.5)   # Clip the gradients
            # clip_grad_norm_(self.models[key+"_vqdecoder"].parameters(), 0.5)   # Clip the gradients
            # clip_grad_norm_(self.models[key+"_quantizer"].parameters(), 0.5)   # Clip the gradients
            total_loss.backward()
            self.optimizers[key].step()
            # Log
            log_str = "Train | part {:s} | epoch [{:d}/{:d}] | step [{:d}/{:d}] | loss {:.5f}".format(
                key, epoch, self.opt["train"]["num_epochs"], iter, total_iters, total_loss.item())
            try:
                self.writer.add_scalar("Train/loss", total_loss, self.global_step)
            except:
                pass
            for k, val in losses_info.items():
                log_str += " | {:s} {:.5f}".format(k, val.item())
                try:
                    self.writer.add_scalar("Train/{:s}".format(k), val, self.global_step)
                except:
                    pass
            
            if self.global_step % self.opt["train"].get("log_per_step", 50) == 0:
                self.logger.info(log_str)
        self.global_step += 1
        
    def eval_one_step(self, batch, epoch):
        losses = {}
        recons = {}
        for key, x in batch.items():
            if key not in self.opt["train"]["part_to_train"]: continue
            if key == "body":
                y, total_loss, losses_info = self.run_one_step_two_stage(x.detach().to(self.device).float(), key=key)
                losses[key] = losses_info[key+"_g_rc"]
            else:
                y, total_loss, losses_info = self.run_one_step_one_stage(x.detach().to(self.device).float(), key=key)
                losses[key] = losses_info[key+"_g_rc"]
                
            recons[key] = (x[:1], y[:1])    # Only keep one sample
        return recons, losses
    
    def train_one_epoch(self, epoch, loader):
        for key, _ in self.models.items():
            self.models[key].train()
        for batch_id, batch in enumerate(loader):
            self.train_one_step(batch, epoch, batch_id, len(loader))
        for key, _ in self.schedulers.items():
            self.schedulers[key].step()
        
    def eval_one_epoch(self, epoch, loader):
        for key, _ in self.models.items():
            self.models[key].eval()
        recons = {key: [] for key in self.opt["train"]["part_to_train"]}
        losses = {key: 0.0 for key in self.opt["train"]["part_to_train"]}
        for batch_id, batch in enumerate(tqdm(loader, desc="Evaluating")):
            with torch.no_grad():
                rec, loss = self.eval_one_step(batch, epoch)
            for key, val in loss.items(): losses[key] += val.item()
            if batch_id % 100 == 0:
                for key, item in rec.items(): recons[key].append(item)
        
        # Log the evaluation info
        log_str = "Eval | epoch [{:d}/{:d}]".format(epoch, self.opt["train"]["num_epochs"])
        for key, val in losses.items():
            log_str += " | {:s}_loss {:.5f}".format(key, val / len(loader))
            try:
                self.writer.add_scalar("Eval/{:s}_loss".format(key), val, epoch)
            except:
                pass
        self.logger.info(log_str)
        
        # Animate
        if epoch % 5 == 0:
            keys = list(recons.keys())
            num_samples = min(10, len(recons[keys[0]]))
            for i in range(num_samples):
                print("Plotting [{:d}/{:d}]".format(i+1, num_samples))
                gt_input = {key+"_pose": val[i][0].float().to(self.device) for key, val in recons.items()}
                rc_input = {key+"_pose": val[i][1].float().to(self.device) for key, val in recons.items()}
                with torch.no_grad():
                    if len(gt_input) == 3:
                        gt_joints = convert_smplx2joints(self.smplx_model, **gt_input)["joints"].data.cpu().numpy()
                        rc_joints = convert_smplx2joints(self.smplx_model, **rc_input)["joints"].data.cpu().numpy()
                        plot_type = "smplx"
                    elif len(gt_input) == 1:
                        gt_joints = convert_smpl2joints(self.smpl_model, **gt_input)["joints"].data.cpu().numpy()
                        rc_joints = convert_smpl2joints(self.smpl_model, **rc_input)["joints"].data.cpu().numpy()
                        plot_type = "smpl"
                    
                    output_path = os.path.join(self.training_folder, "eval", "B{:03d}".format(epoch))
                temp_output_path = os.path.join(output_path, "temp")
                if not os.path.exists(temp_output_path): os.makedirs(temp_output_path)
                try:
                    render_animation(
                        joints_list=[gt_joints[0], rc_joints[0]], 
                        output_path=output_path, 
                        output_name="{:3d}".format(i), 
                        plot_types=[plot_type, plot_type], 
                        prefixs=["gt", "rc"], 
                        video_type="mp4", fps=20)
                except:
                    print("Failed to animate")
    
    def train(self):
        for epoch in range(self.epoch, self.opt["train"]["num_epochs"], 1):
            self.train_one_epoch(epoch, self.train_loader)
            if epoch % self.opt["train"]["save_per_epoch"] == 0:
                self.save_checkpoints(epoch, "Tokenizer_E%03d.pth" % (epoch))
            # if epoch % self.opt["train"]["eval_per_epoch"] == 0:
            #     self.eval_one_epoch(epoch, self.val_loader)
            # if epoch % self.opt["train"]["reset_per_epoch"] == 0:
            #     self.reset_codebook()
        self.save_checkpoints(epoch, "Tokenizer_final.pth")
    
    
