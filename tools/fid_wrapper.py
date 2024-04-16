import os, sys
sys.path.append(os.getcwd())
import warnings
warnings.filterwarnings('ignore')
import importlib
import copy
import yaml
import argparse
import codecs as cs
from tqdm import tqdm
import numpy as np
from scipy import ndimage
import random
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import linalg
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy import interpolate

from smplx import SMPL
from funcs.logger import setup_logger

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] >= multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()

class FID(object):
    def __init__(self, args, config) -> None:
        self.args = args
        self.opt = config
        self.target_ids = args.target_ids.split(",")
        self.normalize_embed = args.normalize_embed
        self.replications = args.replication
        self.valid_length = args.valid_length
        self.max_seq_length = 200
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = importlib.import_module(
            self.opt["model"]["arch_path"], package="networks").__getattribute__(
                self.opt["model"]["arch_name"])(self.opt["model"])
        checkpoint = torch.load(self.args.model, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)
        
    @staticmethod
    def resize_motion(motion, target_length):
        """
        :param motion: [seq_len, dim] numpy array
        :param target_length 
        """
        inp_length = motion.shape[0]
        num_rotvecs = (motion.shape[1] - 3) // 3
        
        # Slerp the rotvecs
        interp_pose = []
        for i in range(num_rotvecs):
            inp_rotvecs = motion[:, 3+i*3:3+(i+1)*3]    # [N, 3]
            key_rots = R.from_rotvec(inp_rotvecs)
            scale = target_length / inp_length
            step = inp_length / (inp_length-1)
            key_times = np.arange(0, inp_length+1, step, dtype=np.float)
            key_times *= scale
            slerp = Slerp(key_times, key_rots)
            times = np.arange(0, target_length).tolist()
            interp_rots = slerp(times)
            interp_rotvecs = interp_rots.as_rotvec()    # [N, 3]
            interp_pose.append(interp_rotvecs)
        interp_pose = np.concatenate(interp_pose, axis=-1)
            
        # Lerp the trans
        interp_trans = []
        for i in range(3):
            inp_trans = motion[:, i]    # [N]
            scale = target_length / inp_length
            step = inp_length / (inp_length-1)
            x = np.arange(0, inp_length+1, step, dtype=np.float)
            x *= scale
            lerp = interpolate.interp1d(x, inp_trans)
            y = lerp(np.arange(0, target_length))
            interp_trans.append(y)
        interp_trans = np.stack(interp_trans, axis=-1)  # [N, 3]
        
        return np.concatenate([interp_trans, interp_pose], axis=-1)
    
    def pad(self, motion, pad_len):
        if pad_len > 0:
            pad_motion = np.zeros((pad_len, motion.shape[-1]))
            motion = np.concatenate([motion, pad_motion], axis=0)
        elif pad_len < 0:
            motion = motion[:self.max_seq_length]
        return motion
    
    def read_data(self, file, resize=False):
        data = np.load(os.path.join(self.args.motion_dir, file), allow_pickle=True).item()
        try:
            gt_pose = data["gt"]["body"][0].transpose(1, 0).copy()
            pred_pose = data["pred"]["body"][0].transpose(1, 0).copy()
        except:
            gt_pose = data["gt"][0].transpose(1, 0).copy()
            pred_pose = data["motion"][0].transpose(1, 0).copy()
            
        gt_pose = torch.from_numpy(gt_pose).unsqueeze(dim=0).float().to(self.device)
        pred_pose = torch.from_numpy(pred_pose).unsqueeze(dim=0).float().to(self.device)
        return gt_pose[:, :self.valid_length], pred_pose[:, :self.valid_length]
    
    @staticmethod
    def calculate_activation_statistics(activations):
        """
        Params:
        -- activation: num_samples x dim_feat
        Returns:
        -- mu: dim_feat
        -- sigma: dim_feat x dim_feat
        """
        mu = np.mean(activations, axis=0)
        cov = np.cov(activations, rowvar=False)
        return mu, cov

    @staticmethod
    def get_metric_statistics(values, replication_times):
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        conf_interval = 1.96 * std / np.sqrt(replication_times)
        return mean, conf_interval
    
    def reorg_files(self):
        all_motion_files = [f for f in os.listdir(self.args.motion_dir) if ".npy" in f]
        files_reorg = {}
        for file in all_motion_files:
            data = np.load(os.path.join(self.args.motion_dir, file), allow_pickle=True).item()
            filename = data["caption"][0]
            found = False
            for targ_id in self.target_ids:
                if targ_id in filename:
                    found = True
                    break
            if not found:
                continue
            bid, tid = file.split(".")[0].split("_")
            if tid in files_reorg.keys():
                files_reorg[tid].append(file)
            else:
                files_reorg[tid] = [file]
        return files_reorg
    
    def run_one_sequence(self, pose_sequence):
        """
        :param pose_sequence: [1, seq_len, dim]
        """
        seq_length = pose_sequence.size(1)
        embeddings = []
        for i in range(0, seq_length, self.max_seq_length//4):
            inp_pose = pose_sequence[:, i:i+self.max_seq_length]
            inp_length = inp_pose.size(1)
            embed = self.model.encode_motion(
                inp_pose.float().to(self.device), 
                torch.tensor(inp_length).long().view(1).to(self.device))
            if self.normalize_embed:
                embed = F.normalize(embed, dim=-1)
            embeddings.append(embed.data.cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings
    
    def calc_frechet_distance(self, all_gt_embeddings, all_pred_embeddings):
        tid_list = list(all_gt_embeddings.keys())
        gt_motion_embeddinsg = all_gt_embeddings[tid_list[0]]
        gt_mu, gt_cov = self.calculate_activation_statistics(gt_motion_embeddinsg)
        
        fid_scores = []
        for rc_motion_embedding in all_pred_embeddings.values():
            rc_mu, rc_cov = self.calculate_activation_statistics(rc_motion_embedding)
            fid = calculate_frechet_distance(gt_mu, gt_cov, rc_mu, rc_cov)
            fid_scores.append(fid)
        return fid_scores

    def calc_diversity(self, all_pred_embeddings):
        div_scores = []
        for rc_motion_embedding in all_pred_embeddings.values():
            div = calculate_diversity(rc_motion_embedding, diversity_times=300)
            div_scores.append(div)
        return div_scores
    
    def run(self):
        files_reorg = self.reorg_files()
        all_gt_embeddings = {key: None for key in files_reorg.keys()}
        all_pred_embeddings = {key: None for key in files_reorg.keys()}
        for tid, files in files_reorg.items():
            gt_embeddings, pred_embeddings = [], []
            for file in tqdm(files, desc="TID: {:s}".format(tid)):
                gt_pose, pred_pose = self.read_data(file=file, resize=False)
                gt_embeddings.append(self.run_one_sequence(pose_sequence=gt_pose))
                pred_embeddings.append(self.run_one_sequence(pose_sequence=pred_pose))
            gt_embeddings = np.concatenate(gt_embeddings, axis=0)
            pred_embeddings = np.concatenate(pred_embeddings, axis=0)
            all_gt_embeddings[tid] = gt_embeddings
            all_pred_embeddings[tid] = pred_embeddings

        # Calculage FID
        fid_scores = self.calc_frechet_distance(all_gt_embeddings, all_pred_embeddings)
        div_scores = self.calc_diversity(all_pred_embeddings)
        
        # FID
        fid_mean, fid_conf = self.get_metric_statistics(fid_scores, replication_times=len(fid_scores))
        print("FID: Mean: {:.4f}, CInf: {:.4f}".format(fid_mean, fid_conf))
        # Div
        div_mean, div_conf = self.get_metric_statistics(div_scores, replication_times=len(div_scores))
        print("DIV: Mean: {:.4f}, CInf: {:.4f}".format(div_mean, div_conf))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                        default='configs/evaluation/config_text2motion_exp1.yaml',    # HumanML3D
                        help='directory to the config file')
    parser.add_argument('--mode', type=str, 
                        default='vald',  
                        help='choose from [test, vald]')
    parser.add_argument('--model', type=str, 
                        default='logs/evaluation/text2motion/exp1/pretrained-1030/checkpoints/Text2MotionAlign_E100.pth',# HumanML3D
                        help='directory to pretrained motion recognition model')
    parser.add_argument('--motion_dir', type=str, 
                        # default="logs/ude/eval/exp9/output/s2m",
                        default="../UDE2.0/logs/baselines/eval/DSG/output/s2m",
                        help='directory to motion generation samples')
    parser.add_argument('--resize_motion', type=str2bool, default=False, help="")
    parser.add_argument('--replication', type=int, default=1, help="")
    parser.add_argument('--normalize_embed', type=str2bool, default=True, help="")
    parser.add_argument('--target_ids', type=str, default="scott", help="")
    parser.add_argument('--valid_length', type=int, default=600, help="")
    args = parser.parse_args()
    
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)

    Agent = FID(args, config)
    Agent.run()