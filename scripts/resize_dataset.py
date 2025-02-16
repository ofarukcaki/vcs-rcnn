import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

def resize_with_padding(image, target_size):
    """Resize image maintaining aspect ratio and add padding if necessary."""
    h, w = image.shape[:2]
    scale = min(target_size[0]/w, target_size[1]/h)
    
    # Calculate new dimensions and scaling info
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    if len(image.shape) == 3:  # RGB image
        interpolation = cv2.INTER_AREA
    else:  # Mask/annotation
        interpolation = cv2.INTER_NEAREST
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    # Calculate padding
    delta_w = target_size[0] - new_w
    delta_h = target_size[1] - new_h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    # Apply padding
    if len(image.shape) == 3:  # RGB image
        color = [0, 0, 0]
    else:  # Mask/annotation
        color = [0]
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                               cv2.BORDER_CONSTANT, value=color)
    
    padding_info = {
        'scale': scale,
        'top': top,
        'left': left,
        'new_h': new_h,
        'new_w': new_w
    }
    
    return padded, padding_info

def process_annotation(annotation_path, padding_info, output_path):
    """Process a single annotation file (semantic, instance, or visibility mask)."""
    if not os.path.exists(annotation_path):
        return
    
    # Read 16-bit PNG annotation
    annotation = cv2.imread(annotation_path, cv2.IMREAD_UNCHANGED)
    if annotation is None:
        print(f"Warning: Could not read annotation {annotation_path}")
        return
    
    # Resize and pad annotation
    resized_annotation, _ = resize_with_padding(annotation, (padding_info['new_w'], padding_info['new_h']))
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as 16-bit PNG
    cv2.imwrite(output_path, resized_annotation)

def process_split(input_split_dir, output_split_dir, target_size, is_test=False):
    """Process a single split (train, val, or test) of the dataset."""
    input_split_dir = Path(input_split_dir)
    output_split_dir = Path(output_split_dir)
    
    # Create output split directory
    output_split_dir.mkdir(parents=True, exist_ok=True)
    
    # Create images directory
    images_dir = output_split_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    
    # For training and validation splits, create annotation directories
    annotation_types = [
        'semantics',
        'plant_instances',
        'leaf_instances',
        'plant_visibility',
        'leaf_visibility'
    ]
    
    if not is_test:
        for ann_type in annotation_types:
            (output_split_dir / ann_type).mkdir(exist_ok=True)
    
    # Get list of images
    image_files = list((input_split_dir / 'images').glob('*.png'))
    if not image_files:
        print(f"Warning: No images found in {input_split_dir / 'images'}")
        return
    
    for image_file in tqdm(image_files, desc=f"Processing {input_split_dir.name} split"):
        # Get image name without path
        image_name = image_file.name
        
        # Process image
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"Warning: Could not read {image_file}")
            continue
        
        # Resize image and get padding info
        resized_image, padding_info = resize_with_padding(image, target_size)
        
        # Save resized image
        output_image_path = images_dir / image_name
        cv2.imwrite(str(output_image_path), resized_image)
        
        # Process annotations only for train and val splits
        if not is_test:
            for ann_type in annotation_types:
                ann_path = input_split_dir / ann_type / image_name
                output_ann_path = output_split_dir / ann_type / image_name
                process_annotation(str(ann_path), padding_info, str(output_ann_path))

def main():
    parser = argparse.ArgumentParser(
        description='Resize PhenoBench dataset images and annotations while preserving aspect ratio')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Input PhenoBench root directory containing train, val, and test splits')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for resized dataset')
    parser.add_argument('--size', type=int, default=512,
                      help='Target size (both width and height)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                      help='Splits to process (default: train val test)')
    
    args = parser.parse_args()
    
    print(f"Processing PhenoBench dataset from {args.input_dir}")
    print(f"Saving resized dataset to {args.output_dir}")
    print(f"Target size: {args.size}x{args.size}")
    print(f"Processing splits: {', '.join(args.splits)}")
    
    # Process each split
    for split in args.splits:
        input_split_dir = Path(args.input_dir) / split
        output_split_dir = Path(args.output_dir) / split
        
        if not input_split_dir.exists():
            print(f"Warning: Split directory {input_split_dir} does not exist, skipping...")
            continue
            
        process_split(input_split_dir, output_split_dir, 
                     target_size=(args.size, args.size),
                     is_test=(split == 'test'))
    
    print("Dataset processing complete!")

if __name__ == "__main__":
    main()
