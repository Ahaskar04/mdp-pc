"""
Image Viewer - Displays all captured images in a tiled grid layout
Run this after the robot completes its run to view all detected characters
"""

import cv2
import os
import numpy as np
from pathlib import Path

# ---- CONFIG ----
IMAGE_SAVE_DIR = "captured_images"
TILE_WIDTH = 640
TILE_HEIGHT = 480
TILES_PER_ROW = 3
BACKGROUND_COLOR = (50, 50, 50)

def load_images_from_directory(directory):
    """Load all captured images from the directory."""
    image_files = []
    
    if not os.path.exists(directory):
        print(f"❌ Directory not found: {directory}")
        return []
    
    # Get all _raw.jpg files (these have bounding boxes)
    for filename in sorted(os.listdir(directory)):
        if filename.endswith("_raw.jpg"):
            filepath = os.path.join(directory, filename)
            image_files.append(filepath)
    
    return image_files

def extract_info_from_filename(filename):
    """Extract obstacle ID and target ID from filename."""
    # Format: obs{obstacle_id}_target{target_id}_{timestamp}_raw.jpg
    basename = os.path.basename(filename)
    try:
        parts = basename.split('_')
        obstacle_id = parts[0].replace('obs', '')
        target_id = parts[1].replace('target', '')
        return obstacle_id, target_id
    except:
        return "?", "?"

def create_tiled_display(image_paths, tiles_per_row=3):
    """Create a tiled grid display of all images."""
    
    if not image_paths:
        print("❌ No images to display")
        return None
    
    # Load all images
    images = []
    labels = []
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            # Resize to standard tile size
            img_resized = cv2.resize(img, (TILE_WIDTH, TILE_HEIGHT))
            
            # Extract info for label
            obs_id, target_id = extract_info_from_filename(img_path)
            label = f"Obs:{obs_id} → {target_id}"
            
            # Add label to image
            cv2.putText(img_resized, label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            images.append(img_resized)
            labels.append(label)
            print(f"✅ Loaded: {label}")
    
    if not images:
        print("❌ No valid images loaded")
        return None
    
    # Calculate grid dimensions
    num_images = len(images)
    num_rows = (num_images + tiles_per_row - 1) // tiles_per_row
    
    # Create blank canvas
    canvas_width = TILE_WIDTH * tiles_per_row
    canvas_height = TILE_HEIGHT * num_rows
    canvas = np.full((canvas_height, canvas_width, 3), BACKGROUND_COLOR, dtype=np.uint8)
    
    # Place images on canvas
    for idx, img in enumerate(images):
        row = idx // tiles_per_row
        col = idx % tiles_per_row
        
        y_start = row * TILE_HEIGHT
        y_end = y_start + TILE_HEIGHT
        x_start = col * TILE_WIDTH
        x_end = x_start + TILE_WIDTH
        
        canvas[y_start:y_end, x_start:x_end] = img
    
    return canvas

def main():
    print("="*60)
    print("IMAGE RECOGNITION RESULTS VIEWER")
    print("="*60)
    print(f"Loading images from: {IMAGE_SAVE_DIR}\n")
    
    # Load images
    image_paths = load_images_from_directory(IMAGE_SAVE_DIR)
    
    if not image_paths:
        print("\n❌ No captured images found!")
        print(f"   Expected location: {os.path.abspath(IMAGE_SAVE_DIR)}")
        print("   Make sure the robot has completed at least one detection.")
        return
    
    print(f"\n📊 Found {len(image_paths)} image(s)\n")
    
    # Create tiled display
    tiled_image = create_tiled_display(image_paths, tiles_per_row=TILES_PER_ROW)
    
    if tiled_image is None:
        return
    
    # Display
    window_name = f"Image Recognition Results ({len(image_paths)} detections)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, tiled_image)
    
    print("\n" + "="*60)
    print("✅ DISPLAY READY")
    print("="*60)
    print("Press any key to close the viewer")
    print("Press 'S' to save the tiled image")
    print("="*60 + "\n")
    
    # Wait for key press
    while True:
        key = cv2.waitKey(100) & 0xFF
        
        if key == ord('s') or key == ord('S'):
            output_path = "tiled_results.jpg"
            cv2.imwrite(output_path, tiled_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"✅ Saved tiled image to: {output_path}")
        elif key != 255:  # Any other key pressed
            break
    
    cv2.destroyAllWindows()
    print("\n👋 Viewer closed")

if __name__ == "__main__":
    main()