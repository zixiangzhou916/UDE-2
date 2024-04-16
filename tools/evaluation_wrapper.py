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

class RPrecision(object):
    def __init__(self, args, config):
        self.args = args
        self.opt = config
        self.normalize_embed = args.normalize_embed
        self.replications = args.replication
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
    
    def reorg_files(self):
        all_motion_files = [f for f in os.listdir(self.args.motion_dir) if ".npy" in f]
        files_reorg = {}
        for file in all_motion_files:
            # if "T0000" not in file: continue
            bid, tid = file.split(".")[0].split("_")
            if tid in files_reorg.keys():
                files_reorg[tid].append(file)
            else:
                files_reorg[tid] = [file]
        return files_reorg
    
    def pad(self, motion, pad_len):
        if pad_len > 0:
            pad_motion = np.zeros((pad_len, motion.shape[-1]))
            motion = np.concatenate([motion, pad_motion], axis=0)
        elif pad_len < 0:
            motion = motion[:self.max_seq_length]
        return motion
    
    def read_data(self, file, resize=False):
        data = np.load(os.path.join(self.args.motion_dir, file), allow_pickle=True).item()
        text = data["caption"][0]
        try:
            gt_pose = data["gt"]["body"][0].transpose(1, 0).copy()
            pred_pose = data["pred"]["body"][0].transpose(1, 0).copy()
        except:
            gt_pose = data["gt"][0].transpose(1, 0).copy()
            pred_pose = data["motion"][0].transpose(1, 0).copy()
        if resize:
            # Resize the length to target_length
            gt_pose = self.resize_motion(gt_pose, target_length=160)
            pred_pose = self.resize_motion(pred_pose, target_length=160)
            gt_length = gt_pose.shape[0]
            pred_length = pred_pose.shape[0]
            # Pad length to 200
            gt_pose = self.pad(gt_pose, self.max_seq_length-gt_pose.shape[0])
            pred_pose = self.pad(pred_pose, self.max_seq_length-pred_pose.shape[0])
        else:
            gt_length = gt_pose.shape[0]
            pred_length = pred_pose.shape[0]
            # Pad length to 200
            gt_pose = self.pad(gt_pose, self.max_seq_length-gt_pose.shape[0])
            pred_pose = self.pad(pred_pose, self.max_seq_length-pred_pose.shape[0])
            
        gt_pose = torch.from_numpy(gt_pose).unsqueeze(dim=0).permute(0, 2, 1).float().to(self.device)
        pred_pose = torch.from_numpy(pred_pose).unsqueeze(dim=0).permute(0, 2, 1).float().to(self.device)
        return gt_pose, pred_pose, gt_length, pred_length, text
    
    def euclidean_distance_matrix(self, matrix1, matrix2):
        """
        :param matrix1: [N1, D]
        :param matrix2: [N2, D]
        """
        assert matrix1.shape[1] == matrix2.shape[1]
        d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
        d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
        d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
        dists = np.sqrt(d1 + d2 + d3)  # broadcasting
        return dists
    
    def calculate_top_k(self, mat, top_k):
        size = mat.shape[0]
        gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
        bool_mat = (mat == gt_mat)
        correct_vec = False
        top_k_list = []
        for i in range(top_k):
            correct_vec = (correct_vec | bool_mat[:, i])
            top_k_list.append(correct_vec[:, None])
        top_k_mat = np.concatenate(top_k_list, axis=1)
        return top_k_mat
    
    @staticmethod
    def get_metric_statistics(values, replication_times):
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        conf_interval = 1.96 * std / np.sqrt(replication_times)
        return mean, conf_interval
    
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

    def calc_frechet_distance(self):
        tid_list = list(self.motion_embeddings["gt"].keys())
        gt_motion_embeddinsg = self.motion_embeddings["gt"][tid_list[0]]
        gt_mu, gt_cov = self.calculate_activation_statistics(gt_motion_embeddinsg)
        
        fid_scores = []
        for rc_motion_embedding in self.motion_embeddings["rc"].values():
            rc_mu, rc_cov = self.calculate_activation_statistics(rc_motion_embedding)
            fid = calculate_frechet_distance(gt_mu, gt_cov, rc_mu, rc_cov)
            fid_scores.append(fid)
        return fid_scores
    
    def calc_diversity(self):
        div_scores = []
        for rc_motion_embedding in self.motion_embeddings["rc"].values():
            div = calculate_diversity(rc_motion_embedding, diversity_times=300)
            div_scores.append(div)
        return div_scores
    
    def calc_multimodality(self):
        mm_scores = []
        tids = list(self.motion_embeddings["rc"].keys())
        # Re-organize the embeddings
        embeddings_reorg = {}
        for tid in tids:
            rc_motion_embedding = self.motion_embeddings["rc"][tid]
            bids = self.motion_embeddings["bid"][tid]
            for (embed, bid) in zip(rc_motion_embedding, bids):
                if bid in embeddings_reorg.keys():
                    embeddings_reorg[bid].append(embed)
                else:
                    embeddings_reorg[bid] = [embed]
        
        min_num_samples = 10000
        for bid, embeds in embeddings_reorg.items():
            if len(embeds) < len(self.tids): continue
            min_num_samples = min(min_num_samples, len(embeds))
            embeddings_reorg[bid] = np.stack(embeds, axis=0)    # [num_samples, num_dim]
        # Calculate multimodality
        for _ in range(30):
            embeddings_cat = []
            for _, embeds in embeddings_reorg.items():
                if len(embeds) < len(self.tids): continue
                i = random.randint(0, len(embeds) - min_num_samples)
                embeddings_cat.append(embeds[i:i+min_num_samples])
            embeddings_cat = np.stack(embeddings_cat, axis=0)   # [num_batches, num_samples, num_dim]
        
            multimodality = calculate_multimodality(embeddings_cat, multimodality_times=min(min_num_samples, len(self.tids)))
            mm_scores.append(multimodality)
        return mm_scores
    
    def prepare(self):
        self.batches = {}
        self.files = [f for f in os.listdir(self.args.motion_dir) if ".npy" in f]
        self.tids = []
        for file in tqdm(self.files, desc="Preparing batches"):
            # Parse the input file to get the tid
            try:
                base_name = file.split(".")[0]
                tid = base_name.split("_")[1]   # sample id
                bid = base_name.split("_")[0]   # batch id
            except:
                bid = base_name
                tid = "T000"
            self.tids.append(tid)
            
            gt_pose, rc_pose, gt_length, rc_length, _ = self.read_data(file, resize=False)
            
            batch = {
                "rc_pose": rc_pose, 
                "gt_pose": gt_pose, 
                "rc_len": rc_length, 
                "gt_len": gt_length, 
                "bid": bid
            }
            if tid in self.batches.keys():
                self.batches[tid].append(batch)
            else:
                self.batches[tid] = [batch]

        self.tids = list(set(self.tids))

    def calc_embeddings(self):
        self.motion_embeddings = {"gt": {}, "rc": {}, "bid": {}}
        for tid, batches in self.batches.items():
            all_gt_embeddings = []
            all_rc_embeddings = []
            all_bids = []
            for i in tqdm(range(0, len(batches)-32, 32), desc="TID: {:s}".format(tid)):
                batch = batches[i:i+32]
                if len(batch) < 32:
                    continue
                gt_motions = torch.cat([b["gt_pose"] for b in batch], dim=0).permute(0, 2, 1)
                gt_lenghts = np.asarray([b["gt_len"] for b in batch])
                rc_motions = torch.cat([b["rc_pose"] for b in batch], dim=0).permute(0, 2, 1)
                rc_lengths = np.asarray([b["rc_len"] for b in batch])
                all_bids += [b["bid"] for b in batch]
                
                with torch.no_grad():
                    gt_embed = self.model.encode_motion(
                        gt_motions.float().to(self.device), 
                        torch.from_numpy(gt_lenghts).long())
                    rc_embed = self.model.encode_motion(
                        rc_motions.float().to(self.device), 
                        torch.from_numpy(rc_lengths).long())
                # Normalize motion embeddings
                if self.normalize_embed:
                    gt_embed = F.normalize(gt_embed, dim=-1)
                    rc_embed = F.normalize(rc_embed, dim=-1)
                
                all_gt_embeddings.append(gt_embed.data.cpu().numpy())
                all_rc_embeddings.append(rc_embed.data.cpu().numpy())
            
            all_gt_embeddings = np.concatenate(all_gt_embeddings, axis=0)
            all_rc_embeddings = np.concatenate(all_rc_embeddings, axis=0)
            self.motion_embeddings["gt"][tid] = all_gt_embeddings
            self.motion_embeddings["rc"][tid] = all_rc_embeddings
            self.motion_embeddings["bid"][tid] = all_bids
    
    def run_rprecision(self):
        files_reorg = self.reorg_files()
        self.gt_matching_scores = []
        self.pred_matching_scores = []
        self.gt_R_precisions = {k: [] for k in range(5)}
        self.pred_R_precisions = {k: [] for k in range(5)}
        for tid, files in files_reorg.items():
            self.batches = len(files)
            for rid in range(self.replications):
                print("--- TID {:s}, Replication {:d}".format(tid, rid))
                random.shuffle(files)
                
                gt_matching_score_sum = 0.0
                pred_matching_score_sum = 0.0
                gt_top_k_count = 0
                pred_top_k_count = 0
                all_size = 0
                for idx in tqdm(range(0, len(files)-32, 32), desc="TID: {:s}".format(tid)):
                    gt_poses, pred_poses, texts = [], [], []
                    gt_lengths, pred_lengths = [], []
                    for i in range(32):
                        gt_pose, pred_pose, gt_length, pred_length, text = self.read_data(files[idx+i], resize=self.args.resize_motion)
                        gt_poses.append(gt_pose)
                        pred_poses.append(pred_pose)
                        texts.append(text)
                        gt_lengths.append(gt_length)
                        pred_lengths.append(pred_length)
                    gt_poses = torch.cat(gt_poses, dim=0)
                    pred_poses = torch.cat(pred_poses, dim=0)
                    # print('----', gt_poses.shape, pred_poses.shape)

                    with torch.no_grad():
                        t_emb = self.model.encode_text(texts)                               # [1, C]
                        m_emb_gt = self.model.encode_motion(gt_poses.permute(0, 2, 1), lengths=gt_lengths)
                        m_emb_pred = self.model.encode_motion(pred_poses.permute(0, 2, 1), lengths=gt_lengths)
                    
                    dist_mat_gt = self.euclidean_distance_matrix(
                        t_emb.data.cpu().numpy(), m_emb_gt.data.cpu().numpy())
                    dist_mat_pred = self.euclidean_distance_matrix(
                        t_emb.data.cpu().numpy(), m_emb_pred.data.cpu().numpy())
                    # Matching scores
                    gt_matching_score_sum += dist_mat_gt.trace()
                    pred_matching_score_sum += dist_mat_pred.trace()
                    # Top-k
                    argsmax_gt = np.argsort(dist_mat_gt, axis=1)
                    argsmax_pred = np.argsort(dist_mat_pred, axis=1)
                    top_k_mat_gt = self.calculate_top_k(argsmax_gt, top_k=5)
                    top_k_mat_pred = self.calculate_top_k(argsmax_pred, top_k=5)
                    gt_top_k_count += top_k_mat_gt.sum(axis=0)
                    pred_top_k_count += top_k_mat_pred.sum(axis=0)

                    all_size += t_emb.shape[0]

                gt_matching_score = gt_matching_score_sum / all_size
                pred_mathinc_score = pred_matching_score_sum / all_size
                gt_R_precision = gt_top_k_count / all_size
                pred_R_precision = pred_top_k_count / all_size

                self.gt_matching_scores.append(gt_matching_score)
                self.pred_matching_scores.append(pred_mathinc_score)
                for k, val in enumerate(gt_R_precision):
                    self.gt_R_precisions[k].append(val)
                for k, val in enumerate(pred_R_precision):
                    self.pred_R_precisions[k].append(val)
                                        
            # GT matching score
        mean, conf_interval = self.get_metric_statistics(np.array(self.gt_matching_scores), replication_times=self.batches)
        print("GT(Matching Scores) | Mean = {:.5f} | CInterval = {:.5f}".format(mean, conf_interval))
        mean, conf_interval = self.get_metric_statistics(np.array(self.pred_matching_scores), replication_times=self.batches)
        print("GT(Matching Scores) | Mean = {:.5f} | CInterval = {:.5f}".format(mean, conf_interval))
        for k in list(self.gt_R_precisions.keys()):
            gt_mean, gt_conf_interval = self.get_metric_statistics(np.array(self.gt_R_precisions[k]), replication_times=self.batches)
            # print("GT(R-Precision @ {:d}) | Mean = {:.5f} | CInterval = {:.5f}".format(k+1, mean, conf_interval))
            pred_mean, pred_conf_interval = self.get_metric_statistics(np.array(self.pred_R_precisions[k]), replication_times=self.batches)
            # print("Pred(R-Precision @ {:d}) | Mean = {:.5f} | CInterval = {:.5f}".format(k+1, mean, conf_interval))
            print("R-Precision @ {:d} | Mean(GT) = {:.5f}, CInterval(GT) = {:.5f} | Mean(Pred) = {:.5f}, CInterval(Pred) = {:.5f}".format(
                k+1, gt_mean, gt_conf_interval, pred_mean, pred_conf_interval))
    
    def run_features(self):
        self.prepare()
        self.calc_embeddings()
        fid_scores = self.calc_frechet_distance()
        div_scores = self.calc_diversity()
        if len(self.tids) > 1:
            mm_scores = self.calc_multimodality()
        
        # FID
        fid_mean, fid_conf = RPrecision.get_metric_statistics(fid_scores, replication_times=len(fid_scores))
        print("FID: Mean: {:.4f}, CInf: {:.4f}".format(fid_mean, fid_conf))
        # Div
        div_mean, div_conf = RPrecision.get_metric_statistics(div_scores, replication_times=len(div_scores))
        print("DIV: Mean: {:.4f}, CInf: {:.4f}".format(div_mean, div_conf))
        # MM
        if len(self.tids) > 1:
            mm_mean, mm_conf = RPrecision.get_metric_statistics(mm_scores, replication_times=len(mm_scores))
            print("MM: Mean: {:.4f}, CInf: {:.4f}".format(mm_mean, mm_conf))
            
    def run(self):
        if self.args.run_rprecision:
            self.run_rprecision()
        print('=' * 100)
        self.run_features()
    
class FeatureMetrics(RPrecision):
    def __init__(self, args, config):
        super(FeatureMetrics, self).__init__(args=args, config=config)
    
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

    def calc_frechet_distance(self):
        tid_list = list(self.motion_embeddings["gt"].keys())
        gt_motion_embeddinsg = self.motion_embeddings["gt"][tid_list[0]]
        gt_mu, gt_cov = self.calculate_activation_statistics(gt_motion_embeddinsg)
        
        fid_scores = []
        for rc_motion_embedding in self.motion_embeddings["rc"].values():
            rc_mu, rc_cov = self.calculate_activation_statistics(rc_motion_embedding)
            fid = calculate_frechet_distance(gt_mu, gt_cov, rc_mu, rc_cov)
            fid_scores.append(fid)
        return fid_scores
    
    def calc_diversity(self):
        div_scores = []
        for rc_motion_embedding in self.motion_embeddings["rc"].values():
            div = calculate_diversity(rc_motion_embedding, diversity_times=300)
            div_scores.append(div)
        return div_scores
    
    def calc_multimodality(self):
        mm_scores = []
        tids = list(self.motion_embeddings["rc"].keys())
        # Re-organize the embeddings
        embeddings_reorg = {}
        for tid in tids:
            rc_motion_embedding = self.motion_embeddings["rc"][tid]
            bids = self.motion_embeddings["bid"][tid]
            for (embed, bid) in zip(rc_motion_embedding, bids):
                if bid in embeddings_reorg.keys():
                    embeddings_reorg[bid].append(embed)
                else:
                    embeddings_reorg[bid] = [embed]
        
        min_num_samples = 10000
        for bid, embeds in embeddings_reorg.items():
            min_num_samples = min(min_num_samples, len(embeds))
            embeddings_reorg[bid] = np.stack(embeds, axis=0)    # [num_samples, num_dim]
        # Calculate multimodality
        for _ in range(self.replications):
            embeddings_cat = []
            for _, embeds in embeddings_reorg.items():
                i = random.randint(0, len(embeds) - min_num_samples)
                embeddings_cat.append(embeds[i:i+min_num_samples])
            embeddings_cat = np.stack(embeddings_cat, axis=0)   # [num_batches, num_samples, num_dim]
        
            multimodality = calculate_multimodality(embeddings_cat, multimodality_times=min(10, len(self.tids)))
            mm_scores.append(multimodality)
       
        return mm_scores
    
    def prepare(self):
        self.batches = {}
        self.files = [f for f in os.listdir(self.args.motion_dir) if ".npy" in f]
        self.tids = []
        for file in tqdm(self.files, desc="Preparing batches"):
            # Parse the input file to get the tid
            base_name = file.split(".")[0]
            tid = base_name.split("_")[1]   # sample id
            bid = base_name.split("_")[0]   # batch id
            self.tids.append(tid)
            
            gt_pose, rc_pose, gt_length, rc_length, _ = self.read_data(file, resize=False)
            
            batch = {
                "rc_pose": rc_pose, 
                "gt_pose": gt_pose, 
                "rc_len": rc_length, 
                "gt_len": gt_length, 
                "bid": bid
            }
            if tid in self.batches.keys():
                self.batches[tid].append(batch)
            else:
                self.batches[tid] = [batch]

        self.tids = list(set(self.tids))

    def calc_embeddings(self):
        self.motion_embeddings = {"gt": {}, "rc": {}, "bid": {}}
        for tid, batches in self.batches.items():
            all_gt_embeddings = []
            all_rc_embeddings = []
            all_bids = []
            for i in tqdm(range(0, len(batches)-32, 32), desc="TID: {:s}".format(tid)):
                batch = batches[i:i+32]
                if len(batch) < 32:
                    continue
                gt_motions = torch.cat([b["gt_pose"] for b in batch], dim=0).permute(0, 2, 1)
                gt_lenghts = np.asarray([b["gt_len"] for b in batch])
                rc_motions = torch.cat([b["rc_pose"] for b in batch], dim=0).permute(0, 2, 1)
                rc_lengths = np.asarray([b["rc_len"] for b in batch])
                all_bids += [b["bid"] for b in batch]
                
                with torch.no_grad():
                    gt_embed = self.model.encode_motion(
                        gt_motions.float().to(self.device), 
                        torch.from_numpy(gt_lenghts).long())
                    rc_embed = self.model.encode_motion(
                        rc_motions.float().to(self.device), 
                        torch.from_numpy(rc_lengths).long())
                # Normalize motion embeddings
                gt_embed = F.normalize(gt_embed, dim=-1)
                rc_embed = F.normalize(rc_embed, dim=-1)
                
                all_gt_embeddings.append(gt_embed.data.cpu().numpy())
                all_rc_embeddings.append(rc_embed.data.cpu().numpy())
            
            all_gt_embeddings = np.concatenate(all_gt_embeddings, axis=0)
            all_rc_embeddings = np.concatenate(all_rc_embeddings, axis=0)
            self.motion_embeddings["gt"][tid] = all_gt_embeddings
            self.motion_embeddings["rc"][tid] = all_rc_embeddings
            self.motion_embeddings["bid"][tid] = all_bids
    
    def run(self):
        self.prepare()
        self.calc_embeddings()
        fid_scores = self.calc_frechet_distance()
        div_scores = self.calc_diversity()
        if len(self.tids) > 1:
            mm_scores = self.calc_multimodality()
        
        # FID
        fid_mean, fid_conf = RPrecision.get_metric_statistics(fid_scores, replication_times=len(fid_scores))
        print("FID: Mean: {:.4f}, CInf: {:.4f}".format(fid_mean, fid_conf))
        # Div
        div_mean, div_conf = RPrecision.get_metric_statistics(div_scores, replication_times=len(div_scores))
        print("DIV: Mean: {:.4f}, CInf: {:.4f}".format(div_mean, div_conf))
        # MM
        if len(self.tids) > 1:
            mm_mean, mm_conf = RPrecision.get_metric_statistics(mm_scores, replication_times=len(mm_scores))
            print("MM: Mean: {:.4f}, CInf: {:.4f}".format(mm_mean, mm_conf))
    
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
                        # default="../UDE2.0/logs/baselines/eval/UDE2.0/output/t2m",                          # GT-HumanML3D
                        default="logs/ude/eval/exp5/output/t2m",                          # GT-HumanML3D
                        help='directory to motion generation samples')
    parser.add_argument('--resize_motion', type=str2bool, default=False, help="")
    parser.add_argument('--replication', type=int, default=1, help="")
    parser.add_argument('--normalize_embed', type=str2bool, default=False, help="")
    parser.add_argument('--run_rprecision', type=str2bool, default=False, help="")
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
        
    # Calculate the R-Precision
    Agent = RPrecision(args, config)
    Agent.run()
    
    # # Calculate the FID, Div, and MM
    # Agent = FeatureMetrics(args, config)
    # Agent.run()
    
    