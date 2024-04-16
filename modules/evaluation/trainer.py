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
    
class Text2MotionAlignmentTrainer(object):
    def __init__(self, args, opt):
        self.args = args
        self.opt = opt
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.timestep = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        self.training_folder = os.path.join(self.args.training_folder, self.args.training_name, self.timestep)
        if not os.path.exists(self.training_folder): 
            os.makedirs(self.training_folder)
        self.logger = setup_logger('UDE-2', self.training_folder, get_rank(), filename='align_text2motion_log.txt')
        self.writer = SummaryWriter(log_dir=self.training_folder)
        with open(os.path.join(self.training_folder, "config_align_text2motion.yaml"), 'w') as outfile:
            yaml.dump(self.opt, outfile, default_flow_style=False)
        
        self.epoch = 0
        self.global_step = 0
        self.setup_models()
        self.load_checkpoints()
        self.setup_loaders()
        
    def setup_models(self):
        self.model = importlib.import_module(
            self.opt["model"]["arch_path"], package="networks").__getattribute__(
                self.opt["model"]["arch_name"])(self.opt["model"])
        self.model = self.model.to(self.device)
        
        opt_w_decay = []
        opt_wo_decay = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad: continue
            if "center" in n:
                opt_wo_decay.append(p)
            else:
                opt_w_decay.append(p)
        opt_list = [
            {"params": opt_w_decay, "lr": self.opt["train"]["lr"], "weight_decay": self.opt["train"]["weight_decay"]}, 
            {"params": opt_wo_decay, "lr": self.opt["train"]["lr"], "weight_decay": 0.0}
        ]
        self.optimizer = optim.Adam(opt_list, 
                                    lr=self.opt["train"]["lr"], 
                                    weight_decay=self.opt["train"]["weight_decay"])
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.opt["train"]["step_lr"], 
            gamma=self.opt["train"]["gamma"])
        
    def setup_loaders(self):
        self.train_loader, _ = importlib.import_module(
            ".evaluation.dataloader", package="dataloader").__getattribute__(
                "get_dataloader")(self.opt["data"]["dataset"], 
                                        self.opt["data"]["loader"]["train"], 
                                        meta_dir=None)
        self.val_loader, _ = importlib.import_module(
            ".evaluation.dataloader", package="dataloader").__getattribute__(
                "get_dataloader")(self.opt["data"]["dataset"], 
                                        self.opt["data"]["loader"]["vald"], 
                                        meta_dir=None)
                
    def load_checkpoints(self):
        if os.path.exists(self.opt["train"]["checkpoint"]):
            checkpoint = torch.load(self.opt["train"]["checkpoint"], map_location=torch.device("cpu"))
        else:
            checkpoint = None
            
        if checkpoint is not None:
            self.model = load_partial(self.model, checkpoint["model"])
            
    def save_checkpoints(self, epoch, name):
        state = {
            'model': self.model.state_dict(),
            'opt': self.optimizer.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
        }
        save_dir = os.path.join(self.training_folder, "checkpoints")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(state, os.path.join(save_dir, name))
        
    def run_one_step(self, batch, stage="train"):
        motion, caption, lengths = batch
        motion = motion.detach().float().to(self.device)
        caption = list(caption)
        if stage == "train":
            self.optimizer.zero_grad()

        output = self.model(motion, caption, lengths=lengths, decode=True)
        m_emb = output["m_emb"]     # [B, C]
        t_emb = output["t_emb"]     # [B, C]
        t_recon = output["t_recon"]   # [B, T, D]
        m_recon = output["m_recon"]   # [B, T, D]
        t_similarity = output.get("t_similarity", None)
    
        # Calc loss
        # 1. Recon loss
        t_rc_loss = F.l1_loss(t_recon, motion, reduction="mean")
        m_rc_loss = F.l1_loss(m_recon, motion, reduction="mean")
        
        # 2. Cross-domain similarity loss
        m_emb_norm = m_emb / m_emb.norm(dim=-1, keepdim=True)
        t_emb_norm = t_emb / t_emb.norm(dim=-1, keepdim=True)
        cross_similarity = m_emb_norm @ t_emb_norm.t()    # [B, B]
        labels = torch.arange(cross_similarity.size(0)).to(cross_similarity.device)
        cs_loss = nn.CrossEntropyLoss()(cross_similarity, labels)
        
        # 3. Intra-domain similarity loss
        m_similarity = m_emb_norm @ m_emb_norm.t()  # [B, B]
        t_similarity = t_emb_norm @ t_emb_norm.t()  # [B, B]
        is_loss = F.l1_loss(m_similarity, t_similarity, reduction="mean")
        
        losses = {
            "rc": (t_rc_loss + m_rc_loss) * 0.5, 
            # "cs": cs_loss, 
            "is": is_loss
        }
        
        # 4. InfoNCE loss (optional)
        if t_similarity is not None:
            nce_loss = self.model.calc_infonce(
                text_embeddings=t_emb_norm,     # Normalized embeddings
                motion_embeddings=m_emb_norm,   # Normalized embeddings
                text_similarity=t_similarity)
            losses["nce"] = nce_loss
        
        total_loss = 0.0
        for key, val in losses.items():
            total_loss = total_loss + self.opt["losses"][key] * val
        losses["total"] = total_loss
        
        if stage == "train":
            total_loss.backward()
            self.optimizer.step()
                
        return losses
    
    def train_one_epoch(self, epoch, loader):
        self.model.train()
        for batch_id, batch in enumerate(loader):
            losses = self.run_one_step(batch, stage="train")
            log_str = 'Train | epoch [{:d}/{:d}] | step [{:d}/{:d}]'.format(
                epoch + 1, self.opt["train"]["num_epochs"], batch_id + 1, len(loader))
            for key, val in losses.items():
                log_str += " | {:s}(loss) {:.5f}".format(key, val.item())
                self.writer.add_scalar("Train/{:s}(loss)".format(key), val, self.global_step)
            
            if self.global_step % self.opt["train"].get("log_per_step", 100) == 0:
                self.logger.info(log_str)
            
            self.global_step += 1
        self.scheduler.step()
        
    def eval_one_epoch(self, epoch, loader):
        self.model.eval()
        losses = {}
        for batch_id, batch in enumerate(tqdm(loader, desc="Evaluation")):
            output = self.run_one_step(batch, stage="eval")
            for key, val in output.items():
                if key in losses.keys(): 
                    losses[key].append(val.item())
                else: 
                    losses[key] = [val.item()]
        
        log_str = "Eval | epoch [{:d}/{:d}]".format(epoch+1, self.opt["train"]["num_epochs"])
        for key, val in losses.items():
            log_str += " | {:s}(loss) = {:.5f}".format(key, np.mean(val))
            self.writer.add_scalar("Eval/{:s}(loss)".format(key), np.mean(val), epoch)
        self.logger.info(log_str)
        
    def train(self):
        for epoch in range(self.epoch, self.opt["train"]["num_epochs"], 1):
            self.train_one_epoch(epoch, self.train_loader)
            if epoch % self.opt["train"]["eval_per_epoch"] == 0:
                self.eval_one_epoch(epoch, self.val_loader)
            if epoch % self.opt["train"]["save_per_epoch"] == 0:
                self.save_checkpoints(epoch, "Text2MotionAlign_E%03d.pth" % (epoch))
        self.save_checkpoints(epoch, "Text2MotionAlign_final.pth")
    
    