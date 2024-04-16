import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import importlib
from funcs.logger import setup_logger
from funcs.comm_utils import get_rank
# from datetime import datetime
# from tqdm import tqdm
import yaml
import random

from modules.utils.training_utils import *
from modules.utils.evaluation_utils import *
# from networks import smplx_code

class UDEEvaluator(object):
    def __init__(self, args, opt):
        self.args = args
        self.opt = opt
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.output_dir = os.path.join(self.args.eval_folder, self.args.eval_name, "output")
        if not os.path.exists(self.output_dir): 
            os.makedirs(self.output_dir)
        self.logger = setup_logger('UDE-2', self.output_dir, get_rank(), filename='ude_eval_log.txt')
        self.setup_models()
        self.setup_loaders()
        
    def setup_loaders(self):
        self.eval_loader, _ = importlib.import_module(
            ".ude.dataloader", package="dataloader").__getattribute__(
                "get_dataloader")(self.opt["data"]["dataset"], 
                                        self.opt["data"]["loader"]["test"], 
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
    
    def setup_models(self):
        self.models = {}
        # Build VQ-VAE models
        for cat_name, model_confs in self.opt["model"]["vqvae"].items():
            for part_name, part_confs in model_confs.items():
                ckpt_path = self.opt["eval"]["checkpoints"]["vqvae"][cat_name][part_name]
                models = self.build_vqvae_models(part_confs, ckpt_path, part_name, self.device)
                for key, model in models.items():
                    self.models["{:s}_{:s}_{:s}".format(cat_name, part_name, key)] = model
                    self.logger.info("VQVAE {:s}_{:s}_{:s} model built and checkpoint resumed from {:s} successfully".format(
                        cat_name, part_name, key, ckpt_path))
                
        # Build UDE model
        self.models["ude"] = importlib.import_module(self.opt["model"]["ude"]["arch_path"], package="networks").__getattribute__(
            self.opt["model"]["ude"]["arch_name"])(self.opt["model"]["ude"]).to(self.device)
        for name, model in self.models.items():
            if "quantizer" in name:
                new_name = "_".join(name.split("_")[:2])
                self.models["ude"].setup_quantizer(quantizer=model, name=new_name)
        self.logger.info("UDE model built successfully")
        
        # Load pretrained model
        checkpoint = torch.load(self.opt["eval"]["checkpoints"]["ude"], map_location=torch.device("cpu"))
        self.models["ude"].load_state_dict(checkpoint["model"], strict=True)
        self.logger.info("UDE weights loaded from {:s} successfully".format(self.opt["eval"]["checkpoints"]["ude"]))
    
    @torch.no_grad()
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
    def quantize_motion(self, motion, cat_name, part_name, lengths=None, max_seq_length=54):
        """Tokenize the motion sequence to token sequence. 
        Our model is performed on token space.
        :param cat_name: category name, [t2m, a2m, s2m].
        :param part_name: body part name, [body, left, right]
        """
        batch_size = motion.size(0)
        outputs = self.motion_preprocess(motion)
        
        max_seq_length = max_seq_length # We set the maximum tokens length = 52
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
            if pad_len > 0:
                tok = torch.cat([tok, pad_id.repeat(pad_len)], dim=0)
            tokens.append(tok)
            label = tok.clone()
            label[valid_len:] = -100
            labels.append(label)
        
        tokens = torch.stack(tokens, dim=0)
        labels = torch.stack(labels, dim=0)
        return tokens, labels
    
    @torch.no_grad()
    def quantize_motion_long(self, motion, seg_len, cat_name, part_name, lenghts=None):
        batch_size = motion.size(0)
        sos_id = torch.tensor(self.models["ude"].sos_id).view(1, 1).long().to(self.device)
        eos_id = torch.tensor(self.models["ude"].eos_id).view(1, 1).long().to(self.device)
        pad_id = torch.tensor(self.models["ude"].pad_id).view(1, 1).long().to(self.device)
        tokens = []
        labels = []
        for i in range(0, motion.size(1), seg_len):
            inp_motion = self.motion_preprocess(motion[:, i:i+seg_len])
            embed = self.models["{:s}_{:s}_vqencoder".format(cat_name, part_name)](inp_motion["inp"])
            tok = self.models["{:s}_{:s}_quantizer".format(cat_name, part_name)].map2index(embed)
            tok = tok.reshape(1, -1)
            tok += 3
            tokens.append(tok)
            labels.append(tok)
        tokens = torch.cat(tokens, dim=1)
        labels = torch.cat(labels, dim=1)
        tokens = torch.cat([sos_id, tokens, eos_id], dim=1)
        labels = torch.cat([sos_id, labels, eos_id], dim=1)
        return tokens, labels
    
    @torch.no_grad()
    def decode_tokens(self, tokens, cat_name, part_name):
        """
        :param tokens: [batch_size, seq_len]
        """
        # Get discrete latent from tokens
        vq_latents = self.models["{:s}_{:s}_quantizer".format(cat_name, part_name)].get_codebook_entry(tokens-3)    # [B, T, C]
        recon = self.models["{:s}_{:s}_vqdecoder".format(cat_name, part_name)](vq_latents)
        return recon
    
    @torch.no_grad()
    def decode_tokens_long(self, tokens, seg_len, step_size, cat_name, part_name):
        num_tokens = tokens.size(1)
        decoded_outputs = []
        for i in range(0, num_tokens, step_size//4):
            dec_outputs = self.decode_tokens(tokens=tokens[:,i:i+seg_len], cat_name=cat_name, part_name=part_name)
            if part_name == "body":
                if len(decoded_outputs) == 0:
                    decoded_outputs.append(dec_outputs)
                else:
                    init_pose = dec_outputs[:, :1, :3]
                    last_pose = decoded_outputs[-1][:, -1:]
                    offset = init_pose[..., :3] - last_pose[..., :3]    # [B, 1, 3]
                    dec_outputs[..., :3] -= offset
                    decoded_outputs.append(dec_outputs)
            else:
                decoded_outputs.append(dec_outputs)
        
        # Merge whole-body pose
        merged_whole_body_poses = merge_motion_segments(decoded_outputs, step_size)
        return merged_whole_body_poses
    
    @torch.no_grad()
    def apply_inverse_translation(self, inp_motion, init_trans):
        """
        :param inp_motion: [batch_size, seq_len, num_dim]
        :param init_trans: [batch_size, 1, 3]
        """    
        if inp_motion.size(-1) == 12:
            return inp_motion
        offset = inp_motion[:, :1, :3] - init_trans
        motion = inp_motion.clone()
        motion[:, :, :3] -= offset
        return motion
    
    @torch.no_grad()
    def eval_text_to_motion(self, batch, step, total_step, batch_id=0):
        gt_motion = batch["body"].detach().float()
        lengths = batch["length"]
        gt_motion = gt_motion[:, :lengths[0]]
        text = batch["text"]
        # text = random.choice(batch["text_list"])
        # Generation motion tokens
        pred_tokens = self.models["ude"].generate_text_to_motion(
            text=text, max_num_tokens=54, 
            topk=self.args.topk, sas=self.args.use_sas, 
            temperature=self.args.temperature, 
            task="t2m", part="body")
        # Decode motion tokens
        pred_motion = self.decode_tokens(tokens=pred_tokens, cat_name="t2m", part_name="body")
        # Apply translation and orienation
        pred_motion = self.apply_inverse_translation(inp_motion=pred_motion, init_trans=gt_motion[:, :1, :3].to(self.device))
        self.logger.info("[{:d}/{:d}][Text-to-Motion] text: {:s} | seq_len(gt): {:d} | seq_len(pred): {:d}".format(
            step+1, total_step, text[0], gt_motion.size(1) // 4, pred_tokens.size(1)))
        result = {
            "gt": {"body": gt_motion.permute(0, 2, 1).data.cpu().numpy()}, 
            "pred": {"body": pred_motion.permute(0, 2, 1).data.cpu().numpy()}, 
            "caption": text
        }
        return result
    
    @torch.no_grad()
    def eval_music_to_motion(self, batch, step, total_step, batch_id=0):
        audio = batch["audio"].detach().float().to(self.device)
        gt_motion = batch["body"].detach().float().to(self.device)
        lengths = batch["length"]
        name = batch["name"]
        # Tokenize motion and prepare targets
        mot_primitive_length = 40
        tok_primitive_length = mot_primitive_length//4
        gt_tokens, labels = self.quantize_motion(
            motion=gt_motion[:, :mot_primitive_length], 
            cat_name="a2m", part_name="body", 
            lengths=[mot_primitive_length], 
            max_seq_length=54)
        max_num_tokens = int(audio.size(-1) * 30 / 16000) // 4
        # Generate motoin tokens
        primitive_tokens = gt_tokens[:, :tok_primitive_length+1]
        pred_tokens = self.models["ude"].generate_audio_to_motion(
            audio=audio, tokens=primitive_tokens, 
            max_num_tokens=max_num_tokens-tok_primitive_length, 
            topk=self.args.topk, sas=self.args.use_sas, 
            temperature=self.args.temperature, 
            block_size=160, 
            task="a2m", part="body")
        # Decode motion tokens
        pred_motion = self.decode_tokens(tokens=pred_tokens, cat_name="a2m", part_name="body")
        # Apply translation and orienation
        pred_motion = self.apply_inverse_translation(inp_motion=pred_motion, init_trans=gt_motion[:, :1, :3].to(self.device))
        self.logger.info("[{:d}/{:d}][Music-to-Motion] seq_len(music): {:d} | seq_len(pred): {:d}".format(
            step+1, total_step, audio.size(1), pred_tokens.size(1)))
        result = {
            "gt": {"body": gt_motion.permute(0, 2, 1).data.cpu().numpy()}, 
            "pred": {"body": pred_motion.permute(0, 2, 1).data.cpu().numpy()}, 
            "audio": audio[:1].unsqueeze(1).data.cpu().numpy(), 
            "caption": name
        }
        return result
    
    @torch.no_grad()
    def eval_speech_to_motion(self, batch, step, total_step, batch_id=0):
        audio = batch["audio"].detach().float().to(self.device)
        lengths = batch["length"]
        name = batch["name"]
        emotion = batch["emotion"][:, 0].detach().long().to(self.device)
        speaker_id = batch["speaker_id"].detach().long().to(self.device)
        max_num_tokens = int(audio.size(-1) * 30 / 16000) // 4
        # max_num_tokens = min(100, max_num_tokens)   # We fix the maximum number of generated tokens to 100
        gt_motion_dict = {}
        pred_motion_dict = {}
        for part in ["body", "left", "right"]:
            gt_motion = batch[part].detach().float().to(self.device)
            # Tokenize motion and prepare targets
            mot_primitive_length = 40
            tok_primitive_length = mot_primitive_length//4
            gt_tokens, labels = self.quantize_motion(
                motion=gt_motion[:, :160].clone(), 
                cat_name="s2m", part_name=part, 
                lengths=[160], 
                max_seq_length=54)
            
            # """ DEBUG """
            # if not self.args.s2m_decode_long:
            #     gt_tokens, labels = self.quantize_motion(
            #     motion=gt_motion, 
            #     cat_name="s2m", part_name=part, 
            #     lengths=[gt_motion.size(1)]
            # )
            #     # Case 1. use Conv1d decoder
            #     pred_motion = self.decode_tokens(tokens=gt_tokens[:, 1:-1], cat_name="s2m", part_name=part)
            # else:
            #     gt_tokens, _ = self.quantize_motion_long(motion=gt_motion, seg_len=160, cat_name="s2m", part_name=part, lenghts=None)
            #     # Case 2. use Conv1d + Transformer decoder
            #     pred_motion = self.decode_tokens_long(tokens=gt_tokens[:, 1:-1], seg_len=160//4, step_size=32*4, cat_name="s2m", part_name=part)
            
            # pred_motion = self.apply_inverse_translation(inp_motion=pred_motion, init_trans=gt_motion[:, :1, :3].to(self.device))
            # gt_motion_dict[part] = gt_motion.permute(0, 2, 1).data.cpu().numpy()
            # pred_motion_dict[part] = pred_motion.permute(0, 2, 1).data.cpu().numpy()
            # self.logger.info("[{:d}/{:d}][Speech-to-Motion] seq_len(music): {:d} | seq_len(pred): {:d}".format(
            #     step+1, total_step, audio.size(1), pred_motion.size(1)))
            # """ END DEBUG """
            
            # Generate motoin tokens
            primitive_tokens = gt_tokens[:, :tok_primitive_length+1]
            pred_tokens = self.models["ude"].generate_speech_to_motion(
                audio=audio, emotion=emotion, speaker_id=speaker_id, 
                tokens=primitive_tokens, 
                max_num_tokens=max_num_tokens-tok_primitive_length, 
                topk=self.args.topk, sas=self.args.use_sas, 
                temperature=self.args.temperature, 
                block_size=self.args.s2m_block_size, 
                task="s2m", part=part)
            # Decode motion tokens
            if not self.args.s2m_decode_long:
                # Case 1. use Conv1d decoder
                pred_motion = self.decode_tokens(tokens=pred_tokens, cat_name="s2m", part_name=part)
            else:
                # Case 2. use Conv1d + Transformer decoder
                pred_motion = self.decode_tokens_long(tokens=pred_tokens, seg_len=160//4, step_size=32*4, cat_name="s2m", part_name=part)
            # Apply translation and orienation
            pred_motion = self.apply_inverse_translation(inp_motion=pred_motion[:, :gt_motion.size(1)], init_trans=gt_motion[:, :1, :3].to(self.device))
            self.logger.info("[{:d}/{:d}][Speech-to-Motion] seq_len(music): {:d} | seq_len(pred): {:d}".format(
                step+1, total_step, audio.size(1), pred_tokens.size(1)))
            gt_motion_dict[part] = gt_motion.permute(0, 2, 1).data.cpu().numpy()
            pred_motion_dict[part] = pred_motion.permute(0, 2, 1).data.cpu().numpy()
        
        result = {
            "gt": gt_motion_dict, 
            "pred": pred_motion_dict, 
            "audio": audio[:1].unsqueeze(1).data.cpu().numpy(), 
            "caption": name
        }
        return result
            
    def eval(self):
        for key in self.models.keys():
            self.models[key].eval()
        
        """ DEBUG """
        # import pickle
        # from tqdm import tqdm
        # with open("dataloader/ude/fixed_testset.pickle", "rb") as f:
        #     debug_dataset = pickle.load(f)
            
        # for batch_id, batch in enumerate(tqdm(debug_dataset)):
        #     inp_batch = {
        #         "text": [batch["t2m"]["text"]], 
        #         "body": torch.from_numpy(batch["t2m"]["body"])[None].float().to(self.device), 
        #         "length": torch.tensor(batch["t2m"]["body"].shape[0]).view(1).long().to(self.device)
        #     }
        #     for tid in range(self.args.repeat_times):
        #         result = self.eval_text_to_motion(batch=inp_batch, step=batch_id, total_step=len(debug_dataset))
        #         save_results(
        #             result=result, output_dir=os.path.join(self.output_dir, "t2m"), 
        #             batch_id=batch_id, generation_id=tid)
        
        """ REGULAR """
        for batch_id, batch in enumerate(self.eval_loader):
            task = batch["task"][0]
            for tid in range(self.args.repeat_times):
                if task == "t2m" and "t" in self.args.eval_mode:
                    result = self.eval_text_to_motion(batch=batch, step=batch_id, total_step=len(self.eval_loader), batch_id=batch_id)
                    save_results(
                        result=result, output_dir=os.path.join(self.output_dir, task), 
                        batch_id=batch_id, generation_id=tid)
                elif task == "a2m" and "a" in self.args.eval_mode:
                    result = self.eval_music_to_motion(batch=batch, step=batch_id, total_step=len(self.eval_loader), batch_id=batch_id)
                    save_results(
                        result=result, output_dir=os.path.join(self.output_dir, task), 
                        batch_id=batch_id, generation_id=tid)
                elif task == "s2m" and "s" in self.args.eval_mode:
                    result = self.eval_speech_to_motion(batch=batch, step=batch_id, total_step=len(self.eval_loader), batch_id=batch_id)
                    save_results(
                        result=result, output_dir=os.path.join(self.output_dir, task), 
                        batch_id=batch_id, generation_id=tid)
                else:
                    continue
                
            
    
    
    