import os, sys, argparse
sys.path.append(os.getcwd())
import numpy as np
import torch
from tqdm import tqdm
import imageio
import random
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3

from render.utils.paramUtil import *

import warnings
warnings.filterwarnings("ignore")

def motion_temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    # print(motion.shape)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)

def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        #         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    plt.close()
    
def rotate_joints(joints, rot):
    """
    :param joints: [nframes, njoints, 3]
    """
    T, J = joints.shape[:2]
    joints_reshape = np.reshape(joints, newshape=(T*J, 3))
    
    M = rot.as_matrix()   # [3, 3]
    joints_rot = np.matmul(M, joints_reshape.transpose(1, 0)).transpose(1, 0)
    joints_rot = np.reshape(joints_rot, newshape=(T, J, 3))
    return joints_rot

def main(joints, save_path, caption, fps, radius):
    joints = motion_temporal_filter(joints)
    # Rotate the joint
    rot = R.from_euler("x", -90, degrees=True)
    joints = rotate_joints(joints=joints, rot=rot)
    if joints.shape[1] == 24:
        plot_3d_motion(save_path, t2m_kinematic_chain, joints, title=caption, fps=fps, radius=radius)
    else:
        plot_3d_motion(save_path, smplx_kinematic_chain, joints, title=caption, fps=fps, radius=radius)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joints_dir", type=str, default="logs/ude/eval/exp4/joints/t2m", help='motion npy file dir')
    parser.add_argument("--video_dir", type=str, default="logs/ude/eval/exp4/animation/t2m", help='motion npy file dir')
    parser.add_argument("--targ_tids", type=str, default="T0000", help='')
    parser.add_argument("--fps", type=int, default=20, help='')
    parser.add_argument("--sample_rate", type=int, default=1, help='')
    parser.add_argument("--max_length", type=int, default=100, help='')
    args = parser.parse_args()
    
    joints_dir = args.joints_dir
    video_dir = args.video_dir
    targ_tids = args.targ_tids
    targ_tids = targ_tids.split(",")
    
    for tid in targ_tids:
        cur_joints_dir = os.path.join(joints_dir, tid)
        cur_video_dir = os.path.join(video_dir, tid)
        if not os.path.exists(cur_joints_dir):
            continue
        if not os.path.exists(cur_video_dir):
            os.makedirs(cur_video_dir)
        filename_list = [f.split(".")[0] for f in os.listdir(cur_joints_dir) if ".npy" in f]
    
        for i, file in enumerate(filename_list):

            data = np.load(cur_joints_dir+"/"+file+".npy", allow_pickle=True).item()
            captions = data["caption"][0]
            captions = captions.replace(".", "").replace("/", "_")
            words = captions.split(" ")
            name = "_".join(words[:20])
            
            joints = data["pred"]["body"][:args.max_length]
            save_path = os.path.join(cur_video_dir, file+".gif")
            if os.path.exists(save_path):
                continue
            print('---', save_path, joints.shape)
            main(joints, save_path=save_path, caption=name, fps=args.fps, radius=4)