#!/usr/bin/env python3
"""Script to convert PhenoBench annotations to YOLO format annotations.

Assumes that the input dir of PhenoBench follows the \"dataset format\" as specified on the website.
Achtung! If labels already exists in specified output path, the script will skip generating them.
But the images will be overwritten even if they already exist in the output path.
    
    Typical usage example:

    python3 convert_annos2yolo.py --input_dir ./data/PhenoBench --output_dir ./data/phenobench-yolo

"""

import os
import argparse
import shutil

from tqdm import tqdm
import numpy as np
import cv2

__author__ = "Y. Linn Chong"

__maintainer__ = "Y. Linn Chong" 
__email__ = "linn.chong@uni-bonn.de"
__status__ = "Development"


def one_anno(sem_path, id_path, out_dir, is_verbose=False):
    """Converts the labels of a single image
    """
    out_file_path = os.path.join(out_dir, os.path.basename(sem_path).split('.')[0]+".txt")
    if is_verbose:
        print(f"Write to file {out_file_path}")

    try:
        with open(out_file_path, "x") as out_file:
            # Check if files exist
            if not os.path.exists(id_path):
                print(f"Error: Instance mask file does not exist: {id_path}")
                return
            if not os.path.exists(sem_path):
                print(f"Error: Semantic mask file does not exist: {sem_path}")
                return

            id_mask = cv2.imread(id_path, cv2.IMREAD_UNCHANGED)
            if id_mask is None:
                print(f"Error: Failed to read instance mask file: {id_path}")
                return

            sem_mask = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED)
            if sem_mask is None:
                print(f"Error: Failed to read semantic mask file: {sem_path}")
                return

            img_width_x = id_mask.shape[1]
            img_height_y = id_mask.shape[0]
        
            sem_mask[sem_mask == 3] = 1 # partial crop --> crop
            sem_mask[sem_mask == 4] = 2 # partial weed --> weed
        
            plant_ids = np.unique(id_mask)
        
            soil_mask = sem_mask == 0
            id_mask[soil_mask] = 0
        
            for plant_id in plant_ids:
                if plant_id == 0:
                    continue # skip because this is just bg

                # this check was for when the labeller used to do weird stuff
                if not np.any(id_mask==plant_id):
                    print("no semantics found.")
                    print(f"Check files: {sem_path} and {plant_id}")
                    continue

                # create an instance 
                plant_class_mask = sem_mask[id_mask==plant_id]

                # same instance id can be used for different classes
                if len(np.unique(plant_class_mask)) > 1:
                    print("Warning: File {out_file_path} two instances from different classes have the same id.")
                for plant_class in np.unique(plant_class_mask):
                    if plant_class == 0:  # skip if class is soil. this should never happen in the first place
                        continue
                    plant_mask = np.logical_and((id_mask == plant_id ),(sem_mask == plant_class))
                    plant_indices = np.where(plant_mask)
            
                    plant_max_x = plant_indices[1].max()
                    plant_min_x = plant_indices[1].min()
            
                    plant_max_y = plant_indices[0].max()
                    plant_min_y = plant_indices[0].min()
            
                    plant_center_x = int((plant_max_x + plant_min_x)/2)
                    plant_center_y = int((plant_max_y + plant_min_y)/2)
            
                    plant_width_x = plant_max_x - plant_min_x
                    plant_height_y = plant_max_y - plant_min_y
            
                    out_file.write(f"{plant_class} {plant_center_x/img_width_x} {plant_center_y/img_height_y} {plant_width_x/img_width_x} {plant_height_y/img_height_y}\n")
    except FileExistsError:
        print(f"Skipping {out_file_path} because it already exists.")


def convert_one_split(in_dir, out_dir, split_tag, is_verbose=False):
    split_in_dir = os.path.join(in_dir, split_tag)
    sem_dir = os.path.join(split_in_dir, "semantics")
    id_dir = os.path.join(split_in_dir, "plant_instances")

    output_dir = os.path.join(out_dir, split_tag, "labels")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting labels for {split_tag} split. Writing to {output_dir}...")

    # Filter out hidden files
    image_files = [f for f in os.listdir(sem_dir) if not f.startswith('.')]
    
    for img_name in tqdm(image_files):
        sem_path = os.path.join(sem_dir, img_name)
        id_path = os.path.join(id_dir, img_name)
        one_anno(sem_path, id_path, output_dir)

    img_in_dir = os.path.join(split_in_dir, "images")
    img_out_dir = os.path.join(out_dir, split_tag, "images")
    print(f"Copying images over to {img_out_dir}...")
    os.makedirs(img_out_dir, exist_ok=True)
    for filename in tqdm(os.listdir(img_in_dir)):
        shutil.copy2(os.path.join(img_in_dir, filename),
                     os.path.join(img_out_dir,filename)
                     )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog = "convert_annos2yolo",
                description = "This script converts the annotations of PhenoBench to the YOLO format.",
                epilog = "Example usage:\npython3 convert_annos2yolo.py --input_dir ./data/PhenoBench --output_dir ./data/phenobench-yolo"
                )

    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    args = parser.parse_args()

    convert_one_split(
            in_dir=args.input_dir,
            out_dir=args.output_dir, 
            split_tag="train",
            is_verbose=args.verbose
            )

    convert_one_split(
            in_dir=args.input_dir,
            out_dir=args.output_dir, 
            split_tag="val",
            is_verbose=args.verbose
            )
