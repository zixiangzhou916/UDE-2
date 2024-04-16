import os, sys, argparse
sys.path.append(os.getcwd())
import numpy as np
import imageio

if __name__ == "__main__":
    input_dir_left = "logs/ude/eval/exp5/animation/s2m/T0000/left"
    input_dir_right = "logs/ude/eval/exp5/animation/s2m/T0000/right"
    output_dir = "logs/ude/eval/exp5/animation/s2m/T0000/hands"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    files_left = [f for f in os.listdir(input_dir_left) if ".png" in f]
    files_right = [f for f in os.listdir(input_dir_right) if ".png" in f]
    for file in files_left:
        print('---', file)
        if file not in files_right:
            continue
        left_image = np.array(imageio.imread(input_dir_left+"/"+file), dtype=np.uint8)
        right_image = np.array(imageio.imread(input_dir_right+"/"+file), dtype=np.uint8)
        image = np.concatenate([left_image[..., :3], right_image[..., :3]], axis=0)
        imageio.imsave(output_dir + "/" + file, image)