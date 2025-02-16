import os
import cv2
import numpy as np
import click
import yaml
from os.path import join, dirname, abspath

@click.command()
@click.option('-c',
              '--config',
              type=str,
              help='path to the config file',
              required=True)
@click.option('-p',
              '--pred_dir',
              type=str,
              help='directory containing prediction txt files',
              required=True)
@click.option('-o',
              '--out_dir',
              type=str,
              help='output directory for visualizations',
              default='./visualizations')
def main(config, pred_dir, out_dir):
    # Load config to get validation data path
    cfg = yaml.safe_load(open(config))
    val_data_path = cfg['data']['val']
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Process each prediction file
    for pred_file in os.listdir(pred_dir):
        if not pred_file.endswith('.txt'):
            continue
            
        # Load predictions
        pred_path = os.path.join(pred_dir, pred_file)
        predictions = np.loadtxt(pred_path)
        
        # If only one prediction, reshape to 2D array
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(1, -1)
        
        # Get corresponding image path
        img_name = pred_file.replace('.txt', '.png')  # Assuming images are JPG
        img_path = os.path.join(val_data_path, "images", img_name)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found")
            continue
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue
            
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Draw each prediction
        for pred in predictions:
            # Format: [class_id, center_x, center_y, width, height, confidence]
            class_id, center_x, center_y, bbox_width, bbox_height, confidence = pred
            
            # Convert normalized coordinates to pixel coordinates
            center_x = int(center_x * width)
            center_y = int(center_y * height)
            bbox_width = int(bbox_width * width)
            bbox_height = int(bbox_height * height)
            
            # Calculate corner coordinates
            x1 = max(0, int(center_x - bbox_width/2))
            y1 = max(0, int(center_y - bbox_height/2))
            x2 = min(width-1, int(center_x + bbox_width/2))
            y2 = min(height-1, int(center_y + bbox_height/2))
            
            # Draw rectangle
            color = (0, 255, 0) if class_id == 1 else (0, 0, 255)  # Green for class 1, Red for class 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Add confidence score and class name
            class_name = "crop" if class_id == 1 else "weed"
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save visualization
        out_path = os.path.join(out_dir, img_name)
        cv2.imwrite(out_path, img)
        print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
