import os, sys, argparse
sys.path.append(os.getcwd())
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import colorsys

import trimesh
from trimesh import Trimesh
import pyrender
from pyrender.constants import RenderFlags
import torch
import numpy as np
from tqdm import tqdm
import math
from math import factorial
# import cv2
from scipy.spatial.transform import Rotation as R
import imageio
from smplx import SMPL

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
class Animator(object):
    def __init__(self, vertices, color_labels, faces, reorg_pos=False):
        """
        :param vertices: [T, 6890, 3]
        """
        if reorg_pos:
            vertices = self.put_to_origin(vertices=vertices)
            self.circle_x, self.circle_y = self.get_pseudo_circle_trajectory(
                center=(0, 0), radius=1, num_points=len(vertices))
        vertices = self.put_to_ground(vertices=vertices)
        self.MINS = np.min(np.min(vertices, axis=0), axis=0)
        self.MAXS = np.max(np.max(vertices, axis=0), axis=0)
        print(self.MINS, "|", self.MAXS)
        self.vertices = vertices
        self.color_labels = color_labels
        self.faces = faces
        self.reorg_pos = reorg_pos
        
        self.minx = self.MINS[0] - 0.5
        self.maxx = self.MAXS[0] + 0.5
        self.minz = self.MINS[2] - 0.5 
        self.maxz = self.MAXS[2] + 0.5
    
    @staticmethod
    def put_to_origin(vertices):
        offset_x = vertices[0, 0, 0]
        offset_y = vertices[0, 0, 1]
        # offset_z = np.min(vertices[0, :, 2])
        offset_z = vertices[..., 2].min(axis=-1)[:, None]
        vertices[..., 0] -= offset_x
        vertices[..., 1] -= offset_y
        vertices[..., 2] -= offset_z
        return vertices
    
    @staticmethod
    def get_pseudo_circle_trajectory(center, radius, num_points):
        pi = math.pi
        x = [math.cos(2*pi/num_points*x)*radius for x in range(0, num_points+1)]
        y = [math.sin(2*pi/num_points*x)*radius for x in range(0, num_points+1)]
        return np.asarray(x), np.asarray(y)
    
    @staticmethod
    def put_to_ground(vertices):
        offset_z = np.min(vertices[0, :, 2])
        vertices[..., 2] -= offset_z
        return vertices
    
    def run(self, outdir, name, fps, mode):
        if mode == "dynamic":
            self.run_dynamic(outdir=outdir, name=name, fps=fps)
        elif mode == "static":
            self.run_static(outdir=outdir, name=name)
        elif mode == "images":
            self.run_images(outdir=outdir, name=name)
        
    def run_dynamic(self, outdir, name, fps=20):
        """Render motion sequence in mesh format. 
        The output is video.
        """
        frames = self.vertices.shape[0]
        vid = []
        
        traj_range_x = self.MAXS[0]-self.MINS[0]
        traj_range_y = self.MAXS[1]-self.MINS[1]
        traj_center_x = (self.MAXS[0]+self.MINS[0]) / 2
        traj_center_y = (self.MAXS[1]+self.MINS[1]) / 2
        
        plane = trimesh.creation.box(extents=(traj_range_x + 0.5, traj_range_y + 0.5, 0.01))
        plane = pyrender.Mesh.from_trimesh(plane, smooth=False)
        plane_node = pyrender.Node(mesh=plane, translation=np.array([traj_center_x, traj_center_y, 0.0]))
        
        for i in tqdm(range(frames)):
            subdivided = trimesh.remesh.subdivide(self.vertices[i], self.faces)
            mesh = Trimesh(vertices=subdivided[0], faces=subdivided[1])
            
            # base_color = (0.11, 0.53, 0.8, 0.5)
            if self.color_labels is None:
                base_color = (1, 0.706, 0)
            else:
                if self.color_labels[i] == 1:
                    base_color = (1, 0.706, 0)
                else:
                    base_color = (0.11, 0.53, 0.8, 0.5)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.7,
                alphaMode='OPAQUE',
                baseColorFactor=base_color
            )
            mesh_face_color = np.array([base_color]).repeat(mesh.faces.shape[0], axis=0)
            mesh.visual.face_colors = mesh_face_color
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            
            bg_color = [1, 1, 1, 0.8]
            scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))

            sx, sy, tx, ty = [0.75, 0.75, 0, 0.10]

            camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))

            light = pyrender.DirectionalLight(color=[0.8, 0.8, 0.8], intensity=5.0)
            spot_l = pyrender.SpotLight(color=[.5, .1, .1], intensity=10.0, innerConeAngle=np.pi / 16, outerConeAngle=np.pi /12)
            scene = pyrender.Scene(ambient_light=np.array([0.4, 0.4, 0.4, 1.0]))
            scene.add(light)
            scene.add(spot_l)
            
            scene.add(mesh)
            scene.add_node(plane_node)
            
            # c = np.pi / 2            
            c = -np.pi / 6
            
            cam_pose = np.array(
                [[ 1, 0, 0, (self.minx+self.maxx)/2],
                [ 0, np.cos(0), -np.sin(0), self.MINS[1]-2.5],
                [ 0, np.sin(0), np.cos(0), 3.0],
                [ 0, 0, 0, 1]]
            )
            Rot = R.from_euler("X", angles=60, degrees=True).as_matrix()
            cam_pose[:3, :3] = Rot
            scene.add(camera, pose=cam_pose)
            
            # render scene
            r = pyrender.OffscreenRenderer(960, 960)
            color, _ = r.render(scene, flags=RenderFlags.SHADOWS_DIRECTIONAL)
            vid.append(color)
            r.delete()
        
        out = np.stack(vid, axis=0)
        imageio.mimsave(outdir + "/" + name+'.mp4', out, fps=fps)

    def run_static(self, outdir, name):
        """Render motion sequence in mesh format. 
        The output is image.
        """
        frames = self.vertices.shape[0]
        vid = []
        
        traj_range_x = self.MAXS[0]-self.MINS[0]
        traj_range_y = self.MAXS[1]-self.MINS[1]
        traj_center_x = (self.MAXS[0]+self.MINS[0]) / 2
        traj_center_y = (self.MAXS[1]+self.MINS[1]) / 2
        
        plane = trimesh.creation.box(extents=(traj_range_x + 0.5, traj_range_y + 0.5, 0.01))
        # plane_color = np.array([[192/255,192/255,192/255]])
        plane.visual.face_colors = [35,35,35,200]
        plane = pyrender.Mesh.from_trimesh(plane, smooth=False)
        plane_node = pyrender.Node(mesh=plane, translation=np.array([0.0, 0.0, 0.0]))
        
        plane_outer = trimesh.creation.box(extents=(traj_range_x + 2.0, traj_range_y + 2.0, 0.01))
        # plane_outer_color = np.array([[135/255,135/255,135/255]])
        plane_outer.visual.face_colors = [135,135,135,200]
        plane_outer = pyrender.Mesh.from_trimesh(plane_outer, smooth=False)
        plane_outer_node = pyrender.Node(mesh=plane_outer, translation=np.array([0.0, 0.0, -0.01]))
        
        Rs = np.asarray([255] * frames)
        Gs = np.arange(180, 0, (0-180)/frames)
        Bs = np.asarray([0] * frames)
        
        bg_color = [1, 1, 1, 0.8]
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))
        sx, sy, tx, ty = [0.75, 0.75, 0, 0.10]
        # Perspective matrix. The larger the denominator is, the closer the view point is.
        camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
        light = pyrender.DirectionalLight(color=[0.8, 0.8, 0.8], intensity=5.0)
        spot_l = pyrender.SpotLight(color=[.5, .1, .1], intensity=10.0, innerConeAngle=np.pi / 16, outerConeAngle=np.pi /12)
        scene = pyrender.Scene(ambient_light=np.array([0.4, 0.4, 0.4, 1.0]))
        scene.add(light)
        scene.add(spot_l)
        scene.add_node(plane_node)
        scene.add_node(plane_outer_node)
        
        c = +np.pi / 6
        
        Rot = R.from_euler("xz", angles=[60,30], degrees=True).as_matrix()
        Rot2 = R.from_euler("z", angles=-30, degrees=True).as_matrix()
        # Camera local pose.
        # trans = np.array([(self.minx+self.maxx)/2+1, self.MINS[1]-2.5, 0.0])
        trans = np.array([0.0, -4.0, 0.0])
        # print(trans)
        trans = np.matmul(trans, Rot2)
        # print(trans)
        cam_pose = np.array(
            [[ 1, 0, 0, trans[0]],
            [ 0, np.cos(0), -np.sin(0), trans[1]],
            [ 0, np.sin(0), np.cos(0), 3.0],
            [ 0, 0, 0, 1]]
        )
        cam_pose[:3, :3] = Rot 
        scene.add(camera, pose=cam_pose)
        
        steps = frames // 5
        for i in tqdm(range(0, frames, steps)):
            subdivided = trimesh.remesh.subdivide(self.vertices[i], self.faces)
            mesh = Trimesh(vertices=subdivided[0], faces=subdivided[1])
            if self.reorg_pos:
                offset_x = self.circle_x[i]
                offset_y = self.circle_y[i]
            else:
                offset_x = 0.0
                offset_y = 0.0
            base_color = (Rs[i]/255, Gs[i]/255, Bs[i]/255)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.7,
                roughnessFactor=0.9, 
                alphaMode='OPAQUE',
                smooth=True, 
                baseColorFactor=base_color
            )
            mesh_face_color = np.array([base_color]).repeat(mesh.faces.shape[0], axis=0)
            mesh.visual.face_colors = mesh_face_color
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            mesh_node = pyrender.Node(mesh=mesh, translation=np.array([-traj_center_x + offset_x, -traj_center_y + offset_y, 0.0]))
            
            scene.add_node(mesh_node)            
            
        # render scene
        r = pyrender.OffscreenRenderer(960, 960)
        color, _ = r.render(scene, flags=RenderFlags.SHADOWS_DIRECTIONAL)
        imageio.imsave(outdir + "/" + name+'.png', color)

    def run_images(self, outdir, name):
        frames = self.vertices.shape[0]
        vid = []
                
        Rs = np.asarray([255] * frames)
        Gs = np.arange(180, 0, (0-180)/frames)
        Bs = np.asarray([0] * frames)
        
        bg_color = [1, 1, 1, 0.8]
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))
        sx, sy, tx, ty = [0.75, 0.75, 0, 0.10]
        # Perspective matrix. The larger the denominator is, the closer the view point is.
        camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
        light = pyrender.DirectionalLight(color=[0.8, 0.8, 0.8], intensity=5.0)
        spot_l = pyrender.SpotLight(color=[.5, .1, .1], intensity=10.0, innerConeAngle=np.pi / 16, outerConeAngle=np.pi /12)
        scene = pyrender.Scene(ambient_light=np.array([0.4, 0.4, 0.4, 1.0]))
        scene.add(light)
        scene.add(spot_l)
        
        c = +np.pi / 6
        
        Rot = R.from_euler("xz", angles=[60,30], degrees=True).as_matrix()
        Rot2 = R.from_euler("z", angles=-30, degrees=True).as_matrix()
        # Camera local pose.
        # trans = np.array([(self.minx+self.maxx)/2+1, self.MINS[1]-2.5, 0.0])
        trans = np.array([0.0, -1.0, 0.0])
        # print(trans)
        trans = np.matmul(trans, Rot2)
        # print(trans)
        cam_pose = np.array(
            [[ 1, 0, 0, trans[0]],
            [ 0, np.cos(0), -np.sin(0), trans[1]],
            [ 0, np.sin(0), np.cos(0), 1.0],
            [ 0, 0, 0, 1]]
        )
        cam_pose[:3, :3] = Rot 
        scene.add(camera, pose=cam_pose)
        
        steps = frames // 5
        for i in tqdm(range(0, frames, steps)):
            subdivided = trimesh.remesh.subdivide(self.vertices[i], self.faces)
            mesh = Trimesh(vertices=subdivided[0], faces=subdivided[1])
            if self.reorg_pos:
                offset_x = self.circle_x[i]
                offset_y = self.circle_y[i]
            else:
                offset_x = 0.0
                offset_y = 0.0
            base_color = (Rs[i]/255, Gs[i]/255, Bs[i]/255)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.7,
                roughnessFactor=0.9, 
                alphaMode='OPAQUE',
                smooth=True, 
                baseColorFactor=base_color
            )
            mesh_face_color = np.array([base_color]).repeat(mesh.faces.shape[0], axis=0)
            mesh.visual.face_colors = mesh_face_color
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            mesh_node = pyrender.Node(mesh=mesh, translation=np.array([0.0, 0.0, 0.0]))
            
            scene.add_node(mesh_node)       
            
            # render scene
            r = pyrender.OffscreenRenderer(960, 960)
            color, _ = r.render(scene, flags=RenderFlags.SHADOWS_DIRECTIONAL)
            imageio.imsave(outdir + "/" + name+'_{:05d}.png'.format(i), color)
            
            scene.remove_node(mesh_node)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", type=str, default="logs/ude/eval/exp3/mesh/t2m", help='motion npy file dir')
    parser.add_argument("--video_dir", type=str, default="logs/ude/eval/exp3/animation/t2m", help='motion npy file dir')
    parser.add_argument("--run_mode", type=str, default="dynamic", help='1. dynamic, 2. static')
    parser.add_argument("--targ_tids", type=str, default="T0000", help='')
    parser.add_argument("--reorg_position", type=str2bool, default=False, help='')
    parser.add_argument("--fps", type=int, default=20, help='')
    parser.add_argument("--sample_rate", type=int, default=1, help='')
    parser.add_argument("--max_length", type=int, default=100, help='')
    args = parser.parse_args()
    
    # smpl_mode = SMPL(model_path="./networks/smpl", gender="NEUTRAL", batch_size=1)
    # faces = smpl_mode.faces
    
    mesh_dir = args.mesh_dir
    video_dir = args.video_dir
    targ_tids = args.targ_tids
    targ_tids = targ_tids.split(",")
    
    for tid in targ_tids:
        cur_mesh_dir = os.path.join(mesh_dir, tid)
        cur_video_dir = os.path.join(video_dir, tid)
        if not os.path.exists(cur_video_dir):
            os.makedirs(cur_video_dir)
        filename_list = [f.split(".")[0] for f in os.listdir(cur_mesh_dir) if ".npy" in f]
    
        for i, file in enumerate(filename_list):

            data = np.load(cur_mesh_dir+"/"+file+".npy", allow_pickle=True).item()
            captions = data["caption"][0]
            captions = captions.replace(".", "").replace("/", "_")
            words = captions.split(" ")
            name = "_".join(words[:20])
            
            for key in data["pred"].keys():
                pass
            
                vertices = data["pred"][key]
                faces = data["faces"][key]
                color_labels = data.get("color_labels", None)
            
                cur_part_video_dir = os.path.join(cur_video_dir, key)
                if not os.path.exists(cur_part_video_dir):
                    os.makedirs(cur_part_video_dir)
                    
                fmt = ".mp4" if args.run_mode == "dynamic" else ".png"
                if os.path.exists(os.path.join(cur_part_video_dir, name+fmt)):
                    print("Rendering [{:d}/{:d}] | Caption: {:s} | Done".format(i+1, len(filename_list), name))
                    continue
                
                print("Rendering [{:d}/{:d}] | Caption: {:s}".format(i+1, len(filename_list), captions))
                if key == "body":
                    animator = Animator(
                        vertices=vertices[:args.max_length][::args.sample_rate], 
                        color_labels=color_labels[:args.max_length][::args.sample_rate] if color_labels is not None else None, 
                        faces=faces, 
                        reorg_pos=args.reorg_position)
                    animator.run(outdir=cur_part_video_dir, name=name, mode=args.run_mode, fps=args.fps / args.sample_rate)
                else:
                    animator = Animator(
                        vertices=vertices[:args.max_length][::args.sample_rate], 
                        color_labels=color_labels[:args.max_length][::args.sample_rate] if color_labels is not None else None, 
                        faces=faces, 
                        reorg_pos=True)
                    animator.run(outdir=cur_part_video_dir, name=name, mode="images", fps=args.fps / args.sample_rate)