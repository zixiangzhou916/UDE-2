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

""" Motion Auto-Encoder Trainer """
class AutoEncoderTrainer(object):
    def __init__(self, args, opt):
        self.args = args
        self.opt = opt
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.timestep = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        self.training_folder = os.path.join(self.args.training_folder, self.args.training_name, self.timestep)
        if not os.path.exists(self.training_folder): 
            os.makedirs(self.training_folder)
        self.logger = setup_logger('UDE-2', self.training_folder, get_rank(), filename='vae_log.txt')
        # self.writer = SummaryWriter(log_dir=self.training_folder)
        with open(os.path.join(self.training_folder, "config_vae.yaml"), 'w') as outfile:
            yaml.dump(self.opt, outfile, default_flow_style=False)
        
        self.epoch = 0
        self.global_step = 0
        
        self.setup_models()
        self.load_checkpoints()
        self.setup_loaders()
        self.setup_human_prior_model()
        
    def setup_models(self):
        
        self.models = {}
        self.models["encoder"] = importlib.import_module(
            self.opt["model"]["encoder"]["arch_path"], package="networks").__getattribute__(
                self.opt["model"]["encoder"]["arch_name"])(**self.opt["model"]["encoder"])
        self.models["decoder"] = importlib.import_module(
            self.opt["model"]["decoder"]["arch_path"], package="networks").__getattribute__(
                self.opt["model"]["decoder"]["arch_name"])(**self.opt["model"]["decoder"])
        self.models["encoder"] = self.models["encoder"].to(self.device)
        self.models["decoder"] = self.models["decoder"].to(self.device)
                        
        self.optimizer = optim.Adam(list(self.models["encoder"].parameters()) + 
                                    list(self.models["decoder"].parameters()), lr=self.opt["train"]["lr"])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                   step_size=self.opt["train"]["step_lr"], 
                                                   gamma=self.opt["train"]["gamma"])
        
    def setup_loaders(self):
        self.train_loader, _ = importlib.import_module(
            ".perception.dataloader", package="dataloader").__getattribute__(
                "get_dataloader")(self.opt["data"]["dataset"], 
                                        self.opt["data"]["loader"]["train"], 
                                        meta_dir=None)
        self.val_loader, _ = importlib.import_module(
            ".perception.dataloader", package="dataloader").__getattribute__(
                "get_dataloader")(self.opt["data"]["dataset"], 
                                        self.opt["data"]["loader"]["vald"], 
                                        meta_dir=None)
    
    def setup_human_prior_model(self):
        self.smpl_model = smplx_code.create(**smpl_cfg)
        self.smpl_model = self.smpl_model.to(self.device)
    
    def load_checkpoints(self):
        if os.path.exists(self.opt["train"]["checkpoint"]):
            checkpoint = torch.load(self.opt["train"]["checkpoint"], map_location=torch.device("cpu"))
        else:
            checkpoint = None
            
        if checkpoint is not None:
            self.models["encoder"] = load_partial(self.models["encoder"], checkpoint=checkpoint["encoder"], logger=self.logger)
            self.models["decoder"] = load_partial(self.models["decoder"], checkpoint=checkpoint["decoder"], logger=self.logger)
            
    def save_checkpoints(self, epoch, name):
        state = {
            'encoder': self.models["encoder"].state_dict(),
            'decoder': self.models["decoder"].state_dict(),
            'opt': self.optimizer.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
        }
        save_dir = os.path.join(self.training_folder, "checkpoints")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(state, os.path.join(save_dir, name))
        
    def run_one_step(self, batch, stage="train"):
        
        motion, lengths = batch
        motion = motion.detach().float().to(self.device)
        mask = torch.ones(motion.size(0), motion.size(1)).to(self.device)
        for i, l in enumerate(lengths):
            mask[i, l:] = 0
        
        if stage == "train":
            self.optimizer.zero_grad()
        
        # Encode motion
        motion_embedding = self.models["encoder"](motion, mask.bool()).loc
        recon = self.models["decoder"](motion_embedding, lengths=lengths)
        # Calc loss
        loss = F.l1_loss(recon, motion, reduction="mean")
        
        if stage == "train":
            loss.backward()
            self.optimizer.step()
                    
        return recon, loss
    
    def train_one_epoch(self, epoch, loader):
        for key in self.models.keys():
            self.models[key].train()
        
        for batch_id, batch in enumerate(loader):
            _, loss = self.run_one_step(batch=batch, stage="train")
            log_str = 'Train | epoch [{:d}/{:d}] | step [{:d}/{:d}] | loss {:.5f}'.format(
                epoch + 1, self.opt["train"]["num_epochs"], batch_id + 1, len(loader), loss.item())
            if self.global_step % self.opt["train"].get("log_per_step", 50) == 0:
                self.logger.info(log_str)
            self.global_step += 1
        self.scheduler.step()
    
    def eval_one_epoch(self, epoch, loader):
        for key in self.models.keys():
            self.models[key].eval()
                
        losses = []
        recons = []
        for batch_id, batch in enumerate(loader):
            lengths = batch[1]
            with torch.no_grad():
                recon, loss = self.run_one_step(batch=batch, stage="eval")
            losses.append(loss)
            if batch_id % 100 == 0:
                recons.append([batch[0][:1, :lengths[0]], recon[:1, :lengths[0]]])
        
        loss = torch.stack(losses, dim=0)
        avg_loss = torch.mean(loss)
        log_str = 'Vald | epoch [{:d}/{:d}] | loss {:.5f}'.format(
                epoch + 1, self.opt["train"]["num_epochs"], avg_loss.item())
        self.logger.info(log_str)
        
        # Animate
        if epoch % 5 == 0:
            keys = list(recons)
            num_samples = min(10, len(recons))
            for i in range(num_samples):
                print("Plotting [{:d}/{:d}]".format(i+1, num_samples))
                gt_input = {"body_pose": recons[i][0].float().to(self.device)}
                rc_input = {"body_pose": recons[i][1].float().to(self.device)}
                with torch.no_grad():
                    gt_joints = convert_smpl2joints(self.smpl_model, **gt_input)["joints"].data.cpu().numpy()
                    rc_joints = convert_smpl2joints(self.smpl_model, **rc_input)["joints"].data.cpu().numpy()
                    
                output_path = os.path.join(self.training_folder, "eval", "B{:03d}".format(epoch))
                temp_output_path = os.path.join(output_path, "temp")
                if not os.path.exists(temp_output_path): os.makedirs(temp_output_path)
                try:
                    render_animation(
                        joints_list=[gt_joints[0], rc_joints[0]], 
                        output_path=output_path, 
                        output_name="{:3d}".format(i), 
                        plot_types=["smpl", "smpl"], 
                        prefixs=["gt", "rc"], 
                        video_type="mp4", fps=20)
                except:
                    print("Failed to animate")
                    
    def train(self):
        for epoch in range(self.epoch, self.opt["train"]["num_epochs"], 1):
            self.train_one_epoch(epoch, self.train_loader)
            if epoch % self.opt["train"].get("save_per_epoch", 1) == 0:
                self.save_checkpoints(epoch, "MotionVAE_E%03d.pth" % (epoch))
            if epoch % self.opt["train"].get("eval_per_epoch", 1) == 0:
                self.eval_one_epoch(epoch, self.val_loader)
        self.save_checkpoints(epoch, "MotionVAE_final.pth")
        