import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import time
from collections import deque

# ---- CONFIG ----
MODEL = "runs/detect/aug_21/weights/best(125epochs).pt"
SAVE_DIR = "detected_images_test"
os.makedirs(SAVE_DIR, exist_ok=True)

# Detection parameters
CENTER_THRESHOLD = 0.35  # Only accept detections in center 35% of frame
SIZE_BUFFER_LENGTH = 10  # Track last 10 frame sizes
SIZE_DECREASE_THRESHOLD = 3  # If size decreases for 3 consecutive frames, capture!

# ---- Load YOLO model ----
print("Loading YOLO model...")
model = YOLO(MODEL)
print("✅ Model loaded successfully!")

# Store detected images for tiling
DETECTED_IMAGES = []  # List of tuples: (detection_count, target_name, annotated_image)
detection_counter = 0

# Track size progression
size_history = deque(maxlen=SIZE_BUFFER_LENGTH)
best_detection = None  # Store the best (largest) detection so far
tracking_active = False

def get_best_centered_detection(boxes, img_width, img_height):
    """
    Get the best detection that is:
    1. Near the center of the frame
    2. Has high confidence
    3. Is reasonably large
    
    Returns: (best_idx, box_area) or (None, 0)
    """
    if not boxes or len(boxes) == 0:
        return None, 0
    
    img_center_x = img_width / 2
    img_center_y = img_height / 2
    
    best_score = -1
    best_idx = None
    best_area = 0
    
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
            continue  # Skip peripheral detections
        
        # Calculate area
        area = (x2 - x1) * (y2 - y1)
        
        # Centrality score (closer to center = better)
        centrality_score = 1 - (dist_x + dist_y) / 2
        
        # Combined score: 50% confidence, 40% centrality, 10% size
        size_score = area / (img_width * img_height)
        combined_score = (0.5 * confidence + 
                         0.4 * centrality_score + 
                         0.1 * size_score)
        
        if combined_score > best_score:
            best_score = combined_score
            best_idx = idx
            best_area = area
    
    return best_idx, best_area

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
    global detection_counter, size_history, best_detection, tracking_active
    
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
    print("🎥 Camera Test - MAXIMUM SIZE Detection")
    print("="*60)
    print("Instructions:")
    print("  - Move object TOWARD camera (box grows)")
    print("  - When you move AWAY, it captures the largest image!")
    print("  - Yellow box shows the accepted detection zone")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset tracking")
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
            boxes = results[0].boxes
            
            if boxes and len(boxes) > 0:
                # Get best CENTERED detection
                best_idx, box_area = get_best_centered_detection(boxes, img_width, img_height)
                
                if best_idx is not None:
                    top_cls_id = int(boxes.cls[best_idx])
                    target_name = model.names[top_cls_id]
                    confidence = float(boxes.conf[best_idx])
                    
                    tracking_active = True
                    
                    # Track size history
                    size_history.append(box_area)
                    
                    # Store best detection so far
                    if best_detection is None or box_area > best_detection['area']:
                        best_detection = {
                            'area': box_area,
                            'target_name': target_name,
                            'confidence': confidence,
                            'raw_img': raw_img.copy(),
                            'annotated_img': annotated.copy()
                        }
                        print(f"📈 New best: {target_name} area={box_area:.0f}")
                    
                    # Display tracking info
                    info_text = f"Tracking: {target_name} ({confidence:.2f})"
                    cv2.putText(annotated, info_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    if best_detection:
                        status = f"Best area: {best_detection['area']:.0f} | Current: {box_area:.0f}"
                        cv2.putText(annotated, status, (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Check if size is decreasing (we've passed the closest point)
                    if len(size_history) >= SIZE_DECREASE_THRESHOLD:
                        recent_sizes = list(size_history)[-SIZE_DECREASE_THRESHOLD:]
                        is_decreasing = all(recent_sizes[i] > recent_sizes[i+1] 
                                          for i in range(len(recent_sizes)-1))
                        
                        if is_decreasing and best_detection is not None:
                            print(f"\n🎯 SIZE PEAKED! Capturing best image (area={best_detection['area']:.0f})")
                            
                            detection_counter += 1
                            save_detection_image(
                                detection_counter,
                                best_detection['target_name'],
                                best_detection['raw_img'],
                                best_detection['annotated_img']
                            )
                            
                            # Reset for next detection
                            size_history.clear()
                            best_detection = None
                            tracking_active = False
                            print("✅ Ready for next object!\n")
                else:
                    # Detections found but none are centered
                    cv2.putText(annotated, "Detections found but NOT CENTERED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                # No detection - if we were tracking and have a best, save it
                if tracking_active and best_detection is not None and len(size_history) >= 3:
                    print(f"\n⚠️  Lost detection. Saving best captured image (area={best_detection['area']:.0f})")
                    
                    detection_counter += 1
                    save_detection_image(
                        detection_counter,
                        best_detection['target_name'],
                        best_detection['raw_img'],
                        best_detection['annotated_img']
                    )
                    
                    # Reset
                    size_history.clear()
                    best_detection = None
                    tracking_active = False
                    print("✅ Ready for next object!\n")
                
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
            elif key == ord('r'):
                # Reset tracking
                print("\n🔄 Resetting tracking...")
                size_history.clear()
                best_detection = None
                tracking_active = False
    
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