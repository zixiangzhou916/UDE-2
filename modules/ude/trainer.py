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

class UDETrainer(object):
    def __init__(self, args, opt) -> None:
        self.opt = opt
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.timestep = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        self.training_folder = os.path.join(self.args.training_folder, self.args.training_name, self.timestep)
        if not os.path.exists(self.training_folder): 
            os.makedirs(self.training_folder)
        self.logger = setup_logger('UDE-2', self.training_folder, get_rank(), filename='ude_model_log.txt')

        try:
            self.writer = SummaryWriter(log_dir=self.training_folder)
        except:
            pass
        with open(os.path.join(self.training_folder, "config_ude.yaml"), 'w') as outfile:
            yaml.dump(self.opt, outfile, default_flow_style=False)
            
        self.epoch = 0
        self.global_step = 0
        self.setup_loaders()
        self.setup_models()
        self.load_checkpoints()
        
    def setup_loaders(self):
        self.train_loader, self.train_dataset = importlib.import_module(
            ".ude.dataloader", package="dataloader").__getattribute__(
                "get_dataloader")(self.opt["data"]["dataset"], 
                                        self.opt["data"]["loader"]["train"], 
                                        meta_dir=None)
        self.val_loader, _ = importlib.import_module(
            ".ude.dataloader", package="dataloader").__getattribute__(
                "get_dataloader")(self.opt["data"]["dataset"], 
                                        self.opt["data"]["loader"]["vald"], 
                                        meta_dir=None)
                
    def build_vqvae_models(self, model_conf, ckpt_path, part_name, device):
        """We load pretrained VQ-VAE model."""
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        models = {}
        for key, conf in model_conf.items():
            models[key] = importlib.import_module(conf["arch_path"], package="networks").__getattribute__(
                conf["arch_name"])(**conf).to(device)
            models[key].load_state_dict(checkpoint["{:s}_{:s}".format(part_name, key)], strict=True)
        return models
    
    def build_perception_models(self, model_conf, ckpt_path, device):
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        model = importlib.import_module(model_conf["arch_path"], package="networks").__getattribute__(
            model_conf["arch_name"])(**model_conf).to(device)
        model.load_state_dict(checkpoint["encoder"], strict=True)
        return model
    
    def setup_models(self):
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        # Build VQ-VAE models
        for cat_name, model_confs in self.opt["model"]["vqvae"].items():
            for part_name, part_confs in model_confs.items():
                ckpt_path = self.opt["train"]["checkpoints"]["vqvae"][cat_name][part_name]
                models = self.build_vqvae_models(part_confs, ckpt_path, part_name, self.device)
                for key, model in models.items():
                    self.models["{:s}_{:s}_{:s}".format(cat_name, part_name, key)] = model
                    self.logger.info("VQVAE {:s}_{:s}_{:s} model built and checkpoint resumed from {:s} successfully".format(
                        cat_name, part_name, key, ckpt_path))

        # Build Perception models
        if "perception" in self.opt["model"].keys():
            for cat_name, model_conf in self.opt["model"]["perception"].items():
                ckpt_path = self.opt["train"]["checkpoints"]["perception"][cat_name]
                self.models["perception_{:s}".format(cat_name)] = \
                    self.build_perception_models(model_conf, ckpt_path, self.device)
                self.logger.info("Perception {:s} model built and checkpoint resumed from {:s} successfully".format(
                    cat_name, ckpt_path))
                
        # Build UDE model
        self.models["ude"] = importlib.import_module(self.opt["model"]["ude"]["arch_path"], package="networks").__getattribute__(
            self.opt["model"]["ude"]["arch_name"])(self.opt["model"]["ude"]).to(self.device)
        for name, model in self.models.items():
            if "quantizer" in name:
                new_name = "_".join(name.split("_")[:2])
                self.models["ude"].setup_quantizer(quantizer=model, name=new_name)
            if "perception" in name:
                self.models["ude"].setup_motion_encoder(encoder=model)
        self.optimizers["ude"] = optim.Adam(filter(lambda p: p.requires_grad, self.models["ude"].parameters()), 
                                            lr=self.opt["train"]["lr"], 
                                            betas=(0.9, 0.999), 
                                            weight_decay=self.opt["train"].get("weight_decay", 0.001))
        self.schedulers["ude"] = optim.lr_scheduler.StepLR(
            self.optimizers["ude"], 
            step_size=self.opt["train"].get("step_lr", 500), 
            gamma=self.opt["train"].get("gamma", 0.1))
        self.logger.info("UDE model built successfully")
        
    def load_checkpoints(self):
        checkpoint_list = ["ude"]
        for key in checkpoint_list:
            try:
                checkpoint = torch.load(self.opt["train"]["checkpoints"][key], map_location=torch.device("cpu"))
                try:
                    self.models[key] = load_partial(self.models[key], checkpoint["ude_model"])
                except:
                    self.models[key] = load_partial(self.models[key], checkpoint["model"])
                self.logger.info("Model {:s} resumed from {:s} successfully".format(key, self.opt["train"]["checkpoints"][key]))
            except:
                self.logger.info("Model {:s} train from scratch".format(key))
            # try:
            #     self.optimizers[key].load_state_dict(checkpoint["optimizer"])
            #     self.logger.info("Optimizer {:s} resumed from {:s} successfully".format(key, self.opt["train"]["checkpoints"][key]))
            # except:
            #     self.logger.info("Optimizer {:s} initialized".format(key))
                
    def save_checkpoints(self, epoch, name):
        checkpoint_list = ["ude", "discriminator"]
        for key in checkpoint_list:
            if key not in self.models.keys(): continue
            state = {
                "model": self.models[key].state_dict(), 
                "optimizer": self.optimizers[key].state_dict(), 
                "epoch": epoch, "global_step": self.global_step
            }
            save_dir = os.path.join(self.training_folder, "checkpoints")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(state, os.path.join(save_dir, "{:s}_{:s}".format(key, name)))
        
    def motion_preprocess(self, motion):
        """Because our vqvae takes different types of input, we need to preprocess the 
        motion sequence accordingly.
        """
        if motion.size(-1) == 12:
            return {"inp": motion}
        trans = motion[..., :3].clone()
        pose = motion[..., 3:].clone()
        offsets = trans[:, 1:] - trans[:, :-1]
        zero_trans = torch.zeros(motion.size(0), 1, 3).float().to(motion.device)
        offsets = torch.cat([zero_trans, offsets], dim=1)
        inputs = torch.cat([offsets, pose], dim=-1) # root set to offsets relative previous frame
        return {"inp": inputs, "trans": trans}
    
    @torch.no_grad()
    def quantize_motion(self, motion, cat_name, part_name, lengths=None):
        """Tokenize the motion sequence to token sequence. 
        Our model is performed on token space.
        :param cat_name: category name, [t2m, a2m, s2m].
        :param part_name: body part name, [body, left, right]
        """
        # TODO: 
        batch_size = motion.size(0)
        outputs = self.motion_preprocess(motion)
        
        max_seq_length = 54 # We set the maximum tokens length = 52
        sos_id = torch.tensor(self.models["ude"].sos_id).view(1).long().to(self.device)
        eos_id = torch.tensor(self.models["ude"].eos_id).view(1).long().to(self.device)
        pad_id = torch.tensor(self.models["ude"].pad_id).view(1).long().to(self.device)
        tokens = []
        labels = []
        for (x, l) in zip(outputs["inp"], lengths):
            embed = self.models["{:s}_{:s}_vqencoder".format(cat_name, part_name)](x[None, :l])
            tok = self.models["{:s}_{:s}_quantizer".format(cat_name, part_name)].map2index(embed)
            tok = tok.reshape(-1)
            tok += 3    # Because we append <SOS>, <EOS>, <PAD>
            tok = torch.cat([sos_id, tok, eos_id], dim=0)
            valid_len = tok.size(0)
            pad_len = max_seq_length - tok.size(0)
            if pad_id > 0:
                tok = torch.cat([tok, pad_id.repeat(pad_len)], dim=0)
            tokens.append(tok)
            label = tok.clone()
            label[valid_len:] = -100
            labels.append(label)
        
        tokens = torch.stack(tokens, dim=0)
        labels = torch.stack(labels, dim=0)
        return tokens, labels
    
    @torch.no_grad()
    def collect_log_info(self, training_info, epoch, total_epoch, cur_step, total_step, task="t2m"):
        task_map = {"t2m": "Text-to-Motion", "a2m": "Music-to-Motion", "s2m": "Speech-to-Motion"}
        for name, items in training_info.items():
            loss_info = items["loss"].item()
            sem_enh_info_1 = items.get("sem_enh_1", torch.tensor(0.0)).item()
            sem_enh_info_2 = items.get("sem_enh_2", torch.tensor(0.0)).item()
            accuracy_info = items["accuracy"].item()
            pred_token = items["pred_tokens"][0].data.cpu().numpy()
            gt_token = items["target_tokens"][0].data.cpu().numpy()
            log_str = "Epoch: [{:d}/{:d}] | Iter: [{:d}/{:d}] | Task: {:s} | Part: {:s} | loss: {:.4f} | sem_enh_1: {:.4f} | sem_enh_2: {:.4f} | accuracy: {:.3f}%".format(
                epoch, total_epoch, cur_step, total_step, task_map[task], name, loss_info, sem_enh_info_1, sem_enh_info_2, accuracy_info * 100.)
            self.logger.info(log_str)
    
    @torch.no_grad()
    def collect_tensorboard_info(self, training_info, task="t2m"):
        task_map = {"t2m": "Text-to-Motion", "a2m": "Music-to-Motion", "s2m": "Speech-to-Motion"}
        for name, items in training_info.items():
            loss_info = items["loss"]
            sem_enh_info_1 = items.get("sem_enh_1", torch.tensor(0.0))
            sem_enh_info_2 = items.get("sem_enh_2", torch.tensor(0.0))
            accuracy_info = items["accuracy"]
            self.writer.add_scalar("Train/{:s}/{:s}(loss)".format(task_map[task], name), loss_info, self.global_step)
            self.writer.add_scalar("Train/{:s}/{:s}(semantic_enhancement_1)".format(task_map[task], name), sem_enh_info_1, self.global_step)
            self.writer.add_scalar("Train/{:s}/{:s}(semantic_enhancement_2)".format(task_map[task], name), sem_enh_info_2, self.global_step)
            self.writer.add_scalar("Train/{:s}/{:s}(accuracy)".format(task_map[task], name), accuracy_info*100., self.global_step)
    
    @torch.no_grad()
    def collect_monitoring_info(self, training_info, task="t2m"):
        task_map = {"t2m": "Text-to-Motion", "a2m": "Music-to-Motion", "s2m": "Speech-to-Motion"}
        for name, items in training_info.items():
            pred_token = items["pred_tokens"][0].data.cpu().numpy().tolist()
            gt_token = items["target_tokens"][0].data.cpu().numpy().tolist()
            self.logger.info("Train | Task: {:s} | Part: {:s} | Tokens(GT): {:s}".format(
                task_map[task], name, ", ".join(map(str, gt_token))))
            self.logger.info("Train | Task: {:s} | Part: {:s} | Tokens(Pred): {:s}".format(
                task_map[task], name, ", ".join(map(str, pred_token))))
            
    def calculate_accuracy(self, pred, target, ignore_cls):
        acc_mask = pred.eq(target).float()
        valid_mask = target.ne(ignore_cls).float()
        accuracy = acc_mask.sum() / valid_mask.sum()
        return accuracy
    
    def calc_vq_loss_and_accuracy(self, pred_logits, targ_labels):
        """
        :param pred_tokens: [batch_size, seq_len, num_dim]
        :param targ_labels: [batch_size, seq_len]
        """
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        pred_tokens = pred_logits.argmax(dim=-1)
        loss = loss_fn(
            pred_logits.contiguous().view(-1, pred_logits.size(-1)), 
            targ_labels.contiguous().view(-1))
        accuracy = self.calculate_accuracy(
            pred=pred_tokens, target=targ_labels, ignore_cls=-100)
        results = {
            "loss": loss, 
            "accuracy": accuracy, 
            "pred_tokens": pred_tokens, 
            "target_tokens": targ_labels
        }
        return results
    
    def calc_total_loss(self, results, task):
        loss = 0.0
        for name, result_item in results.items():
            loss += result_item.get("loss", 0.0) * self.opt["lambdas"]["tasks"].get(task, 1.0)
            loss += result_item.get("sem_enh_1", 0.0) * self.opt["lambdas"]["tasks"].get(task, 1.0) * \
                self.opt["lambdas"]["semantic_enhancement"].get("sem_enh_1", 1.0)
            loss += result_item.get("sem_enh_2", 0.0) * self.opt["lambdas"]["tasks"].get(task, 1.0) * \
                self.opt["lambdas"]["semantic_enhancement"].get("sem_enh_2", 1.0)
        return loss    
    
    def run_text_to_motion_one_step(self, batch):
        losses_info = {}
        accuracy_info = {}
        motion = batch["body"].detach().to(self.device).float()
        texts = batch["text"]
        lengths = batch["length"].data.cpu().numpy().tolist()
        batch_size = motion.size(0)
                        
        # Tokenize motion and prepare targets
        tokens, labels = self.quantize_motion(
            motion=motion, cat_name="t2m", part_name="body", lengths=lengths)
        # Run the text-to-motion
        pred_logits, glob_emb, seq_emb, cond_embeds_dict = self.models["ude"].text_to_motion(
            text=texts, input_ids=tokens[:, :-1])
        # Calculate the vq loss and vq accuracy
        vq_results = {
            "body": self.calc_vq_loss_and_accuracy(pred_logits=pred_logits["body"], targ_labels=labels[:, 1:])
        }
        # Calculate semantic enhancement loss
        calc_semantic_enhancement = False
        if "semantic_enhancement" in self.opt["train"] and self.opt["train"]["semantic_enhancement"].get("t2m", False):
            calc_semantic_enhancement = True
        if hasattr(self.models["ude"], "motion_encoder") and calc_semantic_enhancement:
            motion_embed = self.models["ude"].get_motion_embedding(motion=motion, lengths=lengths)
            sem_enh_losses = calc_semantic_enhancement_loss(
                cond_embed=cond_embeds_dict["fused_emb"], 
                motion_embed=motion_embed.detach())
            vq_results["body"].update(sem_enh_losses)
        
        return vq_results
    
    def run_music_to_motion_one_step(self, batch):
        losses_info = {}
        accuracy_info = {}
        motion = batch["body"].detach().to(self.device).float()
        audio = batch["audio"].detach().to(self.device).float()
        lengths = batch["length"].data.cpu().numpy().tolist()
        batch_size = motion.size(0)
        
        # Tokenize motion and prepare targets
        tokens, labels = self.quantize_motion(
            motion=motion, cat_name="a2m", part_name="body", lengths=lengths)
        # Run the text-to-motion
        pred_logits, glob_emb, seq_emb, cond_embeds_dict = self.models["ude"].audio_to_motion(
            audio=audio, input_ids=tokens[:, :-1])
        # Calculate the vq loss and vq accuracy
        vq_results = {
            "body": self.calc_vq_loss_and_accuracy(pred_logits=pred_logits["body"], targ_labels=labels[:, 1:])
        }
        # Calculate semantic enhancement loss
        calc_semantic_enhancement = False
        if "semantic_enhancement" in self.opt["train"] and self.opt["train"]["semantic_enhancement"].get("a2m", False):
            calc_semantic_enhancement = True
        if hasattr(self.models["ude"], "motion_encoder") and calc_semantic_enhancement:
            motion_embed = self.models["ude"].get_motion_embedding(motion=motion, lengths=lengths)
            sem_enh_losses = calc_semantic_enhancement_loss(
                cond_embed=cond_embeds_dict["fused_emb"], 
                motion_embed=motion_embed.detach())
            vq_results["body"].update(sem_enh_losses)
        
        return vq_results
    
    def run_speech_to_motion_one_step(self, batch):
        audio = batch["audio"].detach().to(self.device).float()
        emotion = batch["emotion"][:, 0].detach().to(self.device).long()
        ids = batch["speaker_id"].detach().to(self.device).long()
        lengths = batch["length"].data.cpu().numpy().tolist()
        name = batch["name"]
        vq_results = {}
        for part in self.opt["train"].get("part_to_train", ["body"]):
            motion = batch[part].detach().to(self.device).float()
            batch_size = motion.size(0)
            # Tokenize motion and prepare targets
            tokens, labels = self.quantize_motion(
                motion=motion, cat_name="s2m", part_name=part, lengths=lengths)
            # Run the text-to-motion
            pred_logits_dict, glob_emb, seq_emb, cond_embeds_dict = self.models["ude"].speech_to_motion(
                audio=audio, ids=ids, emotion=emotion, 
                input_ids_dict={part: tokens[:, :-1]})
            # Calculate the vq loss and vq accuracy
            vq_results[part] = self.calc_vq_loss_and_accuracy(pred_logits=pred_logits_dict[part], targ_labels=labels[:, 1:])
            # vq_results = {
            #     part: self.calc_vq_loss_and_accuracy(pred_logits=pred_logits_dict[part], targ_labels=labels[:, 1:])
            # }
            # Calculate semantic enhancement loss
            calc_semantic_enhancement = False
            if "semantic_enhancement" in self.opt["train"] and self.opt["train"]["semantic_enhancement"].get("s2m", False):
                calc_semantic_enhancement = True
            if part == "body" and hasattr(self.models["ude"], "motion_encoder") and calc_semantic_enhancement:
                motion_embed = self.models["ude"].get_motion_embedding(motion=motion, lengths=lengths)
                sem_enh_losses = calc_semantic_enhancement_loss(
                    cond_embed=cond_embeds_dict["fused_emb"], 
                    motion_embed=motion_embed.detach())
                vq_results[part].update(sem_enh_losses)
            
        return vq_results
    
    def train_one_step(self, batch, epoch, step, total_step):
        # Clear the gradients
        self.optimizers["ude"].zero_grad()
        # Train each domain
        results = {}
        for cat_name, batch_per_cat in batch.items():
            if cat_name == "t2m":
                output = self.run_text_to_motion_one_step(batch=batch_per_cat)
                results[cat_name] = output
            elif cat_name == "a2m":
                output = self.run_music_to_motion_one_step(batch=batch_per_cat)
                results[cat_name] = output
            elif cat_name == "s2m":
                output = self.run_speech_to_motion_one_step(batch=batch_per_cat)
                results[cat_name] = output
        
        total_loss = 0.0
        for cat_name, item_per_cat in results.items():
            # loss_per_cat = 0.0
            # for part_name, item_per_part in item_per_cat.items():
            #     loss_per_cat += item_per_part.get("loss", 0.0) * self.opt["lambdas"].get(cat_name, 1.0)
            loss_per_cat = self.calc_total_loss(results=item_per_cat, task=cat_name)
            total_loss += loss_per_cat
        # Calculate the gradients and update the parameters
        total_loss.backward()
        self.optimizers["ude"].step()
        # Log
        for cat_name, item_per_cat in results.items():
            if self.global_step % self.opt["train"].get("log_per_step", 10) == 0:
                self.collect_log_info(
                    item_per_cat, epoch=epoch, 
                    total_epoch=self.opt["train"]["num_epochs"], 
                    cur_step=step, total_step=total_step, task=cat_name)
            if self.global_step % self.opt["train"].get("monitor_per_step", 100) == 0:
                self.collect_monitoring_info(item_per_cat, task=cat_name)
            self.collect_tensorboard_info(item_per_cat, task=cat_name)
            
    def train_one_epoch(self, epoch, loader):
        
        for key in self.models.keys():
            if key in self.opt["train"]["model_to_train"]:
                self.models[key].train()
            else:
                self.models[key].eval()
            
        total_batch = len(loader)
        for iter, batch in enumerate(loader):
            self.train_one_step(batch=batch, epoch=epoch, step=iter, total_step=len(loader))
            self.global_step += 1
        
        for key in self.schedulers.keys():
            self.schedulers[key].step() # Update the optimizer
    
    def train(self):
        for epoch in range(self.epoch, self.opt["train"]["num_epochs"], 1):
            self.train_one_epoch(epoch, self.train_loader)
            if epoch % self.opt["train"]["save_per_epoch"] == 0:
                self.save_checkpoints(epoch, "UDE_E{:04d}.pth".format(epoch))
            # if epoch % self.opt["train"]["eval_per_epoch"] == 0:
            #     self.eval_one_epoch(epoch, self.val_loader, animate=True if epoch % 10 == 0 else False)
        self.save_checkpoints(epoch, "UDE_final.pth")
            
            
            