#!/usr/bin/env python3
"""
Data augmentation script for PhenoBench dataset.
This script applies various augmentations to images and their corresponding annotations
while maintaining the proper PhenoBench dataset structure.
"""

import os
import random
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import cv2

class DataAugmenter:
    """Handles data augmentation for PhenoBench dataset images and annotations."""
    
    def __init__(self, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2),
                 saturation_range=(0.8, 1.2), rotation_range=(-30, 30)):
        """Initialize augmentation parameters."""
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.rotation_range = rotation_range

    def load_data(self, data_path: Path, image_name: str) -> dict:
        """Load image and corresponding annotations."""
        data = {}
        
        # Load RGB image
        image_path = data_path / 'images' / image_name
        if not image_path.exists():
            return data
        
        data['image'] = Image.open(image_path).convert('RGB')
        
        # Load semantic segmentation
        semantic_path = data_path / 'semantics' / image_name
        if semantic_path.exists():
            data['semantics'] = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED)
            
        # Load instance segmentation
        instance_path = data_path / 'plant_instances' / image_name
        if instance_path.exists():
            data['instances'] = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED)
            
        return data

    def save_data(self, data: dict, output_path: Path, image_name: str):
        """Save augmented data maintaining PhenoBench structure."""
        # Create output directories
        (output_path / 'images').mkdir(parents=True, exist_ok=True)
        if 'semantics' in data:
            (output_path / 'semantics').mkdir(parents=True, exist_ok=True)
        if 'instances' in data:
            (output_path / 'plant_instances').mkdir(parents=True, exist_ok=True)
            
        # Save RGB image
        data['image'].save(output_path / 'images' / image_name)
        
        # Save semantic segmentation
        if 'semantics' in data:
            cv2.imwrite(str(output_path / 'semantics' / image_name), data['semantics'])
            
        # Save instance segmentation
        if 'instances' in data:
            cv2.imwrite(str(output_path / 'plant_instances' / image_name), data['instances'])

    def augment_color(self, image: Image.Image) -> Image.Image:
        """Apply color augmentation to image."""
        image = image.copy()
        
        # Apply random color augmentations
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(
                random.uniform(*self.brightness_range))
        if random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(
                random.uniform(*self.contrast_range))
        if random.random() < 0.5:
            image = ImageEnhance.Color(image).enhance(
                random.uniform(*self.saturation_range))
            
        return image

    def augment_rotation(self, image, angle: float):
        """Rotate image or mask by given angle."""
        return image.rotate(angle, expand=True, resample=Image.BILINEAR)

    def augment_flip(self, image, flip_type: str):
        """Flip image horizontally or vertically."""
        if isinstance(image, np.ndarray):
            # Handle numpy arrays (masks)
            if flip_type == 'horizontal':
                return cv2.flip(image, 1)
            elif flip_type == 'vertical':
                return cv2.flip(image, 0)
        else:
            # Handle PIL images
            if flip_type == 'horizontal':
                return image.transpose(Image.FLIP_LEFT_RIGHT)
            elif flip_type == 'vertical':
                return image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def augment_data(self, data: dict) -> dict:
        """Apply augmentations to both image and annotations."""
        aug_data = {
            'image': data['image'].copy(),
            'semantics': data['semantics'].copy() if 'semantics' in data else None,
            'instances': data['instances'].copy() if 'instances' in data else None
        }
        
        # Color augmentation (only applies to RGB image)
        aug_data['image'] = self.augment_color(aug_data['image'])
        
        # Geometric augmentations (apply to both image and masks)
        if random.random() < 0.3:  # 30% chance of rotation
            angle = random.uniform(*self.rotation_range)
            aug_data['image'] = self.augment_rotation(aug_data['image'], angle)
            if aug_data['semantics'] is not None:
                aug_data['semantics'] = self.augment_rotation(
                    Image.fromarray(aug_data['semantics']), angle)
                aug_data['semantics'] = np.array(aug_data['semantics'])
            if aug_data['instances'] is not None:
                aug_data['instances'] = self.augment_rotation(
                    Image.fromarray(aug_data['instances']), angle)
                aug_data['instances'] = np.array(aug_data['instances'])
        
        if random.random() < 0.5:  # 50% chance of flip
            flip_type = random.choice(['horizontal', 'vertical'])
            aug_data['image'] = self.augment_flip(aug_data['image'], flip_type)
            if aug_data['semantics'] is not None:
                aug_data['semantics'] = self.augment_flip(aug_data['semantics'], flip_type)
            if aug_data['instances'] is not None:
                aug_data['instances'] = self.augment_flip(aug_data['instances'], flip_type)
                
        return aug_data

def main():
    parser = argparse.ArgumentParser(description='Data Augmentation for PhenoBench Dataset')
    parser.add_argument('--train_dir', type=str, required=True,
                      help='Directory containing PhenoBench dataset structure')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save augmented data')
    parser.add_argument('--num_augmentations', type=int, default=1,
                      help='Number of augmentations per image')
    parser.add_argument('--num_images', type=int, default=None,
                      help='Number of images to process (default: all)')
    args = parser.parse_args()

    # Convert paths to Path objects
    train_dir = Path(args.train_dir)
    output_dir = Path(args.output_dir)

    if not train_dir.exists():
        print(f"Training directory {train_dir} does not exist.")
        return

    # Initialize augmenter
    augmenter = DataAugmenter()

    # Get list of images
    image_dir = train_dir / 'images'
    if not image_dir.exists():
        print(f"Images directory {image_dir} does not exist.")
        return

    image_files = list(image_dir.glob('*.png'))
    print(f"Found {len(image_files)} images in {image_dir}")

    # Sample images if specified
    if args.num_images:
        image_files = random.sample(image_files, min(args.num_images, len(image_files)))

    print(f"Processing {len(image_files)} images...")

    # Process each image
    for image_path in image_files:
        try:
            # Load image and annotations
            data = augmenter.load_data(train_dir, image_path.name)
            if not data:
                print(f"Could not load data for {image_path.name}, skipping...")
                continue

            # Apply augmentations
            for i in range(args.num_augmentations):
                aug_data = augmenter.augment_data(data)
                aug_name = f"{image_path.stem}_aug_{i}{image_path.suffix}"
                augmenter.save_data(aug_data, output_dir, aug_name)
                print(f"Created augmentation {i+1}/{args.num_augmentations} for {image_path.name}")

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue

    print("Data augmentation completed successfully!")

if __name__ == '__main__':
    main()
