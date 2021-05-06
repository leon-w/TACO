import os
import time
import numpy as np
import json
import csv

from model import MaskRCNN
from config import Config
import visualize
import utils

import cv2
import glob
from pathlib import Path
import shutil
from tqdm import tqdm


def annotate_video(model, class_names, video_path_in, video_path_out):

    tmp_dir = f"tmp_annotate_{np.random.randint(100000, 1000000)}"
    os.mkdir(tmp_dir)

    print("Extracting frames:")
    os.system(f"ffmpeg -i {video_path_in} -vf fps=30 {tmp_dir}/frame_%05d.jpeg -v quiet -stats")

    color_map = visualize.random_colors(len(class_names))

    for p in tqdm(glob.glob(f"{tmp_dir}/frame_*.jpeg"), desc="Annotating frames"):

        image = cv2.imread(p)
        r = model.detect([image])[0]

        #print(f"Annotating {p} with {r['class_ids'].shape} labels")

        if r['class_ids'].shape[0] > 0:
            r_fused = utils.fuse_instances(r)
        else:
            r_fused = r

        img = visualize.render_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], color_map=color_map)
        cv2.imwrite(f"{tmp_dir}/annotated_{Path(p).name}", img)
    
    print("Rendering video:")
    os.system(f"ffmpeg -r 30 -i {tmp_dir}/annotated_frame_%05d.jpeg -c:v libx264 -vf fps=30 -pix_fmt yuv420p {video_path_out} -v quiet -stats")

    shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Mask R-CNN trained on TACO on a video file')
    parser.add_argument('--model', required=True, help="Path to weights .h5 file")
    parser.add_argument('--input', required=True, help='Path to input video/image')
    parser.add_argument('--output', default="annotated.mp4", help='output path')

    args = parser.parse_args()

    class_names = ['BG', 'Bottle', 'Bottle cap', 'Can', 'Cigarette', 'Cup', 'Lid', 'Other', 'Plastic bag + wrapper', 'Pop tab', 'Straw']

    # Configurations
    class TacoTestConfig(Config):
        NAME = "taco"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 10
        NUM_CLASSES = len(class_names)
        USE_OBJECT_ZOOM = False
    config = TacoTestConfig()
    #config.display()

    model = MaskRCNN(mode="inference", config=config, model_dir="logs/")
    model.load_weights(args.model, args.model, by_name=True)

    annotate_video(model, class_names, args.input, args.output)