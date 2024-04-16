import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from funcs.logger import setup_logger
from funcs.comm_utils import get_rank
from tqdm import tqdm
import yaml
import random

from modules.utils.training_utils import *
from modules.utils.evaluation_utils import *
# from networks import smplx_code

EMOTION_MAP = {
    "neutral": 0, 
    "happiness": 1, 
    "anger": 2, 
    "sadness": 3, 
    "contempt": 4, 
    "surprise": 5, 
    "fear": 6, 
    "disgust": 7
}

SPEAKER_ID = {
    "wayne": 0, "kieks": 1, "nidal": 2, "zhao": 3, "lu": 4,
    "zhang": 5, "carlos": 6, "jorge": 7, "itoi": 8, "daiki": 9,
    "jaime": 10, "scott": 11, "li": 12, "ayana": 13, "luqi": 14,
    "hailing": 15, "kexin": 16, "goto": 17, "reamey": 18, "yingqing": 19,
    "tiffnay": 20, "hanieh": 21, "solomon": 22, "katya": 23, "lawrence": 24,
    "stewart": 25, "carla": 26, "sophie": 27, "catherine": 28, "miranda": 29, 
    "ChenShuiRuo": 0
}

class DEMO(object):
    def __init__(self, args, opt):
        self.args = args
        self.opt = opt
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.output_dir = os.path.join(self.args.eval_folder, self.args.eval_name, "output")
        if not os.path.exists(self.output_dir): 
            os.makedirs(self.output_dir)
        self.logger = setup_logger('UDE-2', self.output_dir, get_rank(), filename='demo_log.txt')
        self.setup_models()

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
        # TODO: 
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
    
    def generate_motion_from_text(self, input_files):
        import json
        with open(input_files, "r") as f:
            texts = json.load(f)
        
        for bid, text in tqdm(enumerate(texts)):
            for gid in range(self.args.num_generation):
                pred_tokens = self.models["ude"].generate_text_to_motion(
                    text=[text], max_num_tokens=54, 
                    topk=self.args.topk, sas=self.args.use_sas, 
                    temperature=self.args.temperature, 
                    task="t2m", part="body")
                pred_motion = self.decode_tokens(tokens=pred_tokens, cat_name="t2m", part_name="body")
                pred_motion = self.apply_inverse_translation(inp_motion=pred_motion, init_trans=torch.zeros(1,1,3).to(self.device))
                result = {
                    "pred": {"body": pred_motion.permute(0,2,1).data.cpu().numpy()}, 
                    "caption": [text]
                }
                self.logger.info("text: {:s}, frames of generated motion: {:d}".format(text, pred_motion.size(1)))
                np.save(os.path.join(self.output_dir, "B{:03d}_T{:03d}.npy".format(bid, gid)), result)
            
    def generate_motion_from_music(self, input_files):
        import json
        import librosa
        with open(input_files, "r") as f:
            music_list = json.load(f)
        
        for bid, music_file in enumerate(tqdm(music_list)):
            wav_file, samplerate = librosa.load(music_file)
            wav_file = librosa.resample(wav_file, orig_sr=samplerate, target_sr=16000)
            wav_file = torch.from_numpy(wav_file).float().to(self.device).unsqueeze(0)
        
            max_num_tokens = int(wav_file.size(-1) * 30 / 16000) // 4
            primitive_tokens = torch.tensor(self.models["ude"].sos_id).view(1, 1).long().to(self.device)
            for gid in range(self.args.num_generation):
                pred_tokens = self.models["ude"].generate_audio_to_motion(
                    audio=wav_file, 
                    tokens=primitive_tokens, 
                    max_num_tokens=max_num_tokens, 
                    topk=self.args.topk, sas=self.args.use_sas, 
                    temperature=self.args.temperature, 
                    block_size=160,
                    task="a2m", part="body"
                )
                pred_motion = self.decode_tokens(tokens=pred_tokens, cat_name="a2m", part_name="body")
                pred_motion = self.apply_inverse_translation(inp_motion=pred_motion, init_trans=torch.zeros(1,1,3).to(self.device))
                result = {
                    "pred": {"body": pred_motion.permute(0,2,1).data.cpu().numpy()}, 
                    "audio": wav_file.unsqueeze(1).data.cpu().numpy(), 
                    "caption": os.path.split(music_file)[1].split(".")[0]
                }
                self.logger.info("music name: {:s}, music length: {:d}, frames of generated motion: {:d}".format(
                    music_file, int(wav_file.size(1) / samplerate), pred_motion.size(1)))
                np.save(os.path.join(self.output_dir, "B{:03d}_T{:03d}.npy".format(bid, gid)), result)
            
    def generate_motion_from_speech(self, input_files):
        import json
        import librosa
        with open(input_files, "r") as f:
            speech_list = json.load(f)
            
        for bid, speech_file in enumerate(tqdm(speech_list)):
            wav_file, samplerate = librosa.load(speech_file["file"])
            emotion = torch.tensor(EMOTION_MAP[speech_file["emotion"]]).view(1).long().to(self.device)
            speaker_id = torch.tensor(SPEAKER_ID[speech_file["speaker"]]).view(1).long().to(self.device)
            wav_file = librosa.resample(wav_file, orig_sr=samplerate, target_sr=16000)
            wav_file = torch.from_numpy(wav_file).float().to(self.device).unsqueeze(0)
            
            max_num_tokens = int(wav_file.size(-1) * 30 / 16000) // 4
            for gid in range(self.args.num_generation):
                pred_motion_dict = {}
                for body_part in ["body", "left", "right"]:
                    primitive_tokens = torch.tensor(self.models["ude"].sos_id).view(1, 1).long().to(self.device)
                    pred_tokens = self.models["ude"].generate_speech_to_motion(
                        audio=wav_file, 
                        emotion=emotion, 
                        speaker_id=speaker_id, 
                        tokens=primitive_tokens, 
                        max_num_tokens=max_num_tokens, 
                        topk=self.args.topk, sas=self.args.use_sas, 
                        temperature=self.args.temperature, 
                        block_size=160,
                        task="s2m", part=body_part
                    )
                    pred_motion = self.decode_tokens_long(tokens=pred_tokens, seg_len=40, step_size=32*4, cat_name="s2m", part_name=body_part)
                    pred_motion = self.apply_inverse_translation(inp_motion=pred_motion, init_trans=torch.zeros(1,1,3).to(self.device))
                    pred_motion_dict[body_part] = pred_motion.permute(0, 2, 1).data.cpu().numpy()
                
                result = {
                    "pred": pred_motion_dict, 
                    "audio": wav_file.unsqueeze(1).data.cpu().numpy(), 
                    "caption": os.path.split(speech_file["file"])[1].split(".")[0]
                }
                self.logger.info("speech name: {:s}, speech length: {:d}, frames of generated motion: {:d}".format(
                    speech_file["file"], int(wav_file.size(1) / samplerate), pred_motion.size(1)))
                np.save(os.path.join(self.output_dir, "B{:03d}_T{:03d}.npy".format(bid, gid)), result)
    
    def __call__(self, input_files, demo_type="t2m"):
        if demo_type == "t2m":
            self.generate_motion_from_text(input_files)
        if demo_type == "a2m":
            self.generate_motion_from_music(input_files)
        if demo_type == "s2m":
            self.generate_motion_from_speech(input_files)
        
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/ude/config_exp1.yaml', help='path to the config file')
    parser.add_argument('--dataname', type=str, default='HumanML3D', help='name of dataset, choose from [AMASS, AMASS-single, HumanML3D')
    parser.add_argument('--eval_folder', type=str, default='demo_output/', help='path of demo output folder')
    parser.add_argument('--eval_name', type=str, default='s2m', help='name of the demo choose from [t2m, a2m, s2m]')
    parser.add_argument('--input_files', type=str, default='demo_inputs/s2m/speech_list.json', help='')
    parser.add_argument('--num_generation', type=int, default=1, help='number of generation per prompt')
    parser.add_argument('--topk', type=int, default=10, help='')
    parser.add_argument('--temperature', type=float, default=2.0, help='')
    parser.add_argument('--use_sas', type=str2bool, default=True, help="")
    args = parser.parse_args()
    return args    
        
if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        opt = yaml.safe_load(f)
    demo = DEMO(args=args, opt=opt)
    demo(input_files=args.input_files, demo_type=args.eval_name)