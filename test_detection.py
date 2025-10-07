import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import time

# ---- CONFIG ----
MODEL = "runs/detect/aug_21/weights/best(125epochs).pt"
SAVE_DIR = "detected_images_test"
os.makedirs(SAVE_DIR, exist_ok=True)

# Detection parameters
CENTER_THRESHOLD = 0.35  # Only accept detections in center 35% of frame

# ---- Load YOLO model ----
print("Loading YOLO model...")
model = YOLO(MODEL)
print("✅ Model loaded successfully!")

# Store detected images for tiling
DETECTED_IMAGES = []  # List of tuples: (detection_count, target_name, annotated_image)
detection_counter = 0

# Track last detection to avoid duplicates
last_detection_time = 0
DETECTION_COOLDOWN = 2.0  # seconds between detections

def get_best_centered_detection(boxes, img_width, img_height):
    """
    Get the best detection that is:
    1. Near the center of the frame
    2. Has high confidence
    3. Is reasonably large
    
    Returns: best_idx or None
    """
    if not boxes or len(boxes) == 0:
        return None
    
    img_center_x = img_width / 2
    img_center_y = img_height / 2
    
    best_score = -1
    best_idx = None
    
    for idx in range(len(boxes)):
        box = boxes[idx]
        
        # Get box properties
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0].cpu().numpy())
        
        # Calculate box center
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        
        # Calculate normalized distance from center
        dist_x = abs(box_center_x - img_center_x) / img_width
        dist_y = abs(box_center_y - img_center_y) / img_height
        
        # Only consider detections near center
        if dist_x > CENTER_THRESHOLD or dist_y > CENTER_THRESHOLD:
            print(f"   ⚠️  Skipping peripheral detection: {model.names[int(box.cls[0])]} (dist_x={dist_x:.2f}, dist_y={dist_y:.2f})")
            continue  # Skip peripheral detections
        
        # Calculate area (normalized)
        area = ((x2 - x1) * (y2 - y1)) / (img_width * img_height)
        
        # Centrality score (closer to center = better)
        centrality_score = 1 - (dist_x + dist_y) / 2
        
        # Combined score: 50% confidence, 40% centrality, 10% size
        combined_score = (0.5 * confidence + 
                         0.4 * centrality_score + 
                         0.1 * area)
        
        if combined_score > best_score:
            best_score = combined_score
            best_idx = idx
    
    return best_idx

def save_detection_image(detection_id, target_name, raw_img, annotated_img):
    """Save both raw and annotated images locally."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save annotated image with bounding box
    annotated_filename = os.path.join(SAVE_DIR, f"detection_{detection_id}_{target_name}_{timestamp}_annotated.jpg")
    cv2.imwrite(annotated_filename, annotated_img)
    print(f"✅ Saved annotated image: {annotated_filename}")
    
    # Save raw image
    raw_filename = os.path.join(SAVE_DIR, f"detection_{detection_id}_{target_name}_{timestamp}_raw.jpg")
    cv2.imwrite(raw_filename, raw_img)
    print(f"✅ Saved raw image: {raw_filename}")
    
    # Add to global list for tiled display
    DETECTED_IMAGES.append((detection_id, target_name, annotated_img.copy()))
    
    # Update tiled display
    update_tiled_display()

def update_tiled_display():
    """Create a tiled display of all detected images in one window."""
    if not DETECTED_IMAGES:
        return
    
    num_images = len(DETECTED_IMAGES)
    
    # Calculate grid dimensions
    cols = min(3, num_images)  # Max 3 images per row
    rows = (num_images + cols - 1) // cols
    
    # Resize all images to same size for tiling
    tile_size = (320, 240)  # Width x Height for each tile
    tiles = []
    
    for detection_id, target_name, img in DETECTED_IMAGES:
        # Resize image
        resized = cv2.resize(img, tile_size)
        
        # Add label with detection ID and target name
        label = f"#{detection_id}: {target_name}"
        cv2.putText(resized, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        tiles.append(resized)
    
    # Pad with blank images if needed to fill grid
    while len(tiles) < rows * cols:
        blank = np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
        tiles.append(blank)
    
    # Create grid
    grid_rows = []
    for r in range(rows):
        row_tiles = tiles[r * cols:(r + 1) * cols]
        grid_row = np.hstack(row_tiles)
        grid_rows.append(grid_row)
    
    tiled_image = np.vstack(grid_rows)
    
    # Display tiled window
    cv2.imshow("Detected Targets - Tiled View", tiled_image)

def open_camera():
    """Try multiple methods to open camera on macOS."""
    print("\n🔍 Attempting to open camera...")
    
    # Try different camera indices
    for camera_index in [0, 1, 2]:
        print(f"   Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            # Give camera time to initialize
            time.sleep(1)
            
            # Try to read a test frame
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Camera opened successfully on index {camera_index}")
                return cap
            else:
                cap.release()
    
    # Try with different backends on macOS
    print("   Trying AVFoundation backend...")
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        time.sleep(1)
        ret, frame = cap.read()
        if ret and frame is not None:
            print("✅ Camera opened with AVFoundation backend")
            return cap
        else:
            cap.release()
    
    return None

def main():
    global detection_counter, last_detection_time
    
    # Open camera with retry logic
    cap = open_camera()
    
    if cap is None:
        print("\n❌ Error: Could not open any camera")
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure no other app is using the camera")
        print("   2. Check System Preferences > Security & Privacy > Camera")
        print("   3. Grant camera permissions to Terminal/Python")
        print("   4. Try running: 'brew install opencv' if using Homebrew")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n" + "="*60)
    print("🎥 Camera Test - CENTERED Image Detection & Saving")
    print("="*60)
    print("Instructions:")
    print("  - Point camera at objects to detect")
    print("  - Only CENTER detections will be saved")
    print("  - Yellow box shows the accepted detection zone")
    print("  - Press 'q' to quit")
    print("  - Press 's' to manually save current detection")
    print("  - Detections auto-save with 2-second cooldown")
    print("="*60 + "\n")
    
    # Warm up camera - discard first few frames
    print("🔥 Warming up camera...")
    for _ in range(10):
        cap.read()
    print("✅ Camera ready!\n")
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print(f"⚠️  Warning: Failed to grab frame #{frame_count}")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Keep a copy of raw image
            raw_img = frame.copy()
            img_height, img_width = frame.shape[:2]
            
            # Run YOLO inference
            results = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)
            
            # Get annotated image with bounding boxes
            annotated = results[0].plot()
            
            # Draw center region (yellow box) for reference
            center_x = img_width // 2
            center_y = img_height // 2
            roi_w = int(img_width * CENTER_THRESHOLD)
            roi_h = int(img_height * CENTER_THRESHOLD)
            cv2.rectangle(annotated, 
                        (center_x - roi_w, center_y - roi_h),
                        (center_x + roi_w, center_y + roi_h),
                        (0, 255, 255), 2)  # Yellow box showing center region
            
            # Check if any detections
            current_time = datetime.now().timestamp()
            boxes = results[0].boxes
            
            if boxes and len(boxes) > 0:
                # Get best CENTERED detection (ignores periphery)
                best_idx = get_best_centered_detection(boxes, img_width, img_height)
                
                if best_idx is not None:
                    top_cls_id = int(boxes.cls[best_idx])
                    target_name = model.names[top_cls_id]
                    confidence = float(boxes.conf[best_idx])
                    
                    # Display detection info on frame
                    info_text = f"CENTERED: {target_name} ({confidence:.2f})"
                    cv2.putText(annotated, info_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Auto-save if cooldown period has passed
                    if current_time - last_detection_time > DETECTION_COOLDOWN:
                        detection_counter += 1
                        print(f"\n🎯 Detection #{detection_counter}: {target_name} (confidence: {confidence:.2f})")
                        save_detection_image(detection_counter, target_name, raw_img, annotated)
                        last_detection_time = current_time
                else:
                    # Detections found but none are centered
                    cv2.putText(annotated, "Detections found but NOT CENTERED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                # Show "No detection" message
                cv2.putText(annotated, "No detection", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display live feed
            cv2.imshow("Live Camera Feed", annotated)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Exiting...")
                break
            elif key == ord('s'):
                # Manual save
                boxes = results[0].boxes
                if boxes and len(boxes) > 0:
                    best_idx = get_best_centered_detection(boxes, img_width, img_height)
                    if best_idx is not None:
                        detection_counter += 1
                        top_cls_id = int(boxes.cls[best_idx])
                        target_name = model.names[top_cls_id]
                        print(f"\n📸 Manual save #{detection_counter}: {target_name}")
                        save_detection_image(detection_counter, target_name, raw_img, annotated)
                        last_detection_time = current_time
                    else:
                        print("⚠️  Detections found but none are centered")
                else:
                    print("⚠️  No detection to save")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Test complete! Total detections: {detection_counter}")
        print(f"   Total frames processed: {frame_count}")
        print(f"📁 Images saved in: {os.path.abspath(SAVE_DIR)}")

if __name__ == "__main__":
    main()