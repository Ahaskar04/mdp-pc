import socket, struct
import numpy as np
import cv2
from ultralytics import YOLO
import threading
import time
import os
from collections import deque
from datetime import datetime

# ---- CONFIG ----
PI_IP = "192.168.25.1"
PORT_STREAM = 5002
PORT_LOG = 5001
MODEL = "runs/detect/aug_21/weights/best(125epochs).pt"

# Create directory for saved images
SAVE_DIR = "detected_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Detection parameters
CENTER_THRESHOLD = 0.35  # Only accept detections in center 35% of frame
SIZE_BUFFER_LENGTH = 10  # Track last 10 frame sizes
SIZE_DECREASE_THRESHOLD = 3  # If size decreases for 3 consecutive frames, we've passed the closest point

# ---- Load YOLO model ----
model = YOLO(MODEL)

# --- GLOBAL STATE ---
ACTIVE_SNAP_ID = None
SNAP_ARMED = False
DETECTED_OBSTACLE_IDS = set()

# Store detected images for tiling
DETECTED_IMAGES = []

# Track size progression for current obstacle
OBSTACLE_SIZE_HISTORY = deque(maxlen=SIZE_BUFFER_LENGTH)
BEST_DETECTION = None  # Store the best (largest) detection so far

# ---- Utility Functions ----
def recv_exact(sock, n):
    """Receive exactly n bytes or raise an error if connection closes."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf.extend(chunk)
    return bytes(buf)

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

def save_detection_image(obstacle_id, target_name, raw_img, annotated_img):
    """
    Save both raw and annotated images locally.
    Also add to global list for tiled display.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save annotated image with bounding box
    annotated_filename = os.path.join(SAVE_DIR, f"obstacle_{obstacle_id}_{target_name}_{timestamp}_annotated.jpg")
    cv2.imwrite(annotated_filename, annotated_img)
    print(f"✅ Saved annotated image: {annotated_filename}")
    
    # Save raw image (optional - for reference)
    raw_filename = os.path.join(SAVE_DIR, f"obstacle_{obstacle_id}_{target_name}_{timestamp}_raw.jpg")
    cv2.imwrite(raw_filename, raw_img)
    print(f"✅ Saved raw image: {raw_filename}")
    
    # Add to global list for tiled display
    DETECTED_IMAGES.append((obstacle_id, target_name, annotated_img.copy()))
    
    # Update tiled display
    update_tiled_display()

def update_tiled_display():
    """
    Create a tiled display of all detected images in one window.
    """
    if not DETECTED_IMAGES:
        return
    
    num_images = len(DETECTED_IMAGES)
    
    # Calculate grid dimensions
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    # Resize all images to same size for tiling
    tile_size = (320, 240)
    tiles = []
    
    for obstacle_id, target_name, img in DETECTED_IMAGES:
        resized = cv2.resize(img, tile_size)
        label = f"Obs {obstacle_id}: {target_name}"
        cv2.putText(resized, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        tiles.append(resized)
    
    while len(tiles) < rows * cols:
        blank = np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
        tiles.append(blank)
    
    grid_rows = []
    for r in range(rows):
        row_tiles = tiles[r * cols:(r + 1) * cols]
        grid_row = np.hstack(row_tiles)
        grid_rows.append(grid_row)
    
    tiled_image = np.vstack(grid_rows)
    
    cv2.imshow("Detected Targets - Tiled View", tiled_image)
    cv2.waitKey(1)

# ---- RPi COMMAND LISTENER THREAD ----
def rpi_command_listener(log_sock):
    global ACTIVE_SNAP_ID, SNAP_ARMED
    print("RPi Command Listener started.")
    while True:
        try:
            data = log_sock.recv(1024).decode('utf-8').strip()
            if not data:
                continue

            for line in data.splitlines():
                line = line.strip()
                if not line:
                    continue

                if line.startswith("SNAP_ID,"):
                    ACTIVE_SNAP_ID = line.split(',', 1)[1].strip()
                    print(f"\n*** SNAP ID RECEIVED. Active Obstacle ID: {ACTIVE_SNAP_ID} ***")
                elif line == "SNAPREADY":
                    SNAP_ARMED = True
                    print("*** SNAP READY received. Detection ARMED. ***")
                else:
                    print(f"[RPi Log]: {line}")

        except Exception as e:
            print(f"RPi Command Listener Error: {e}")
            break

# ---- MAIN INFERENCE LOOP ----
def main():
    global ACTIVE_SNAP_ID, SNAP_ARMED, DETECTED_OBSTACLE_IDS
    global OBSTACLE_SIZE_HISTORY, BEST_DETECTION

    s_stream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to RPi video stream at {PI_IP}:{PORT_STREAM} ...")
    s_stream.connect((PI_IP, PORT_STREAM))

    s_log = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to RPi log channel at {PI_IP}:{PORT_LOG} ...")
    s_log.connect((PI_IP, PORT_LOG))

    print("All connections established. Starting inference...")

    threading.Thread(target=rpi_command_listener, args=(s_log,), daemon=True).start()

    last_sent_snap_id = None

    try:
        while True:
            # --- Read video stream frame ---
            hdr = recv_exact(s_stream, 8)
            (length,) = struct.unpack(">Q", hdr)
            payload = recv_exact(s_stream, length)
            img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            raw_img = img.copy()
            img_height, img_width = img.shape[:2]

            # --- Run YOLO inference ---
            results = model.predict(source=img, imgsz=640, conf=0.25, verbose=False)

            # --- WAIT FOR MAXIMUM SIZE DETECTION LOGIC ---
            if SNAP_ARMED and ACTIVE_SNAP_ID and ACTIVE_SNAP_ID != last_sent_snap_id:
                
                # Check if already detected
                if ACTIVE_SNAP_ID in DETECTED_OBSTACLE_IDS:
                    print(f"✅ Obstacle {ACTIVE_SNAP_ID} already detected. Skipping...")
                    SNAP_ARMED = False
                    last_sent_snap_id = ACTIVE_SNAP_ID
                    OBSTACLE_SIZE_HISTORY.clear()
                    BEST_DETECTION = None
                    continue
                
                boxes = results[0].boxes
                
                if boxes and len(boxes) > 0:
                    # Get best CENTERED detection
                    best_idx, box_area = get_best_centered_detection(boxes, img_width, img_height)
                    
                    if best_idx is not None:
                        top_cls_id = int(boxes.cls[best_idx])
                        target_name = model.names[top_cls_id]
                        confidence = float(boxes.conf[best_idx])

                        # Track size history
                        OBSTACLE_SIZE_HISTORY.append(box_area)
                        
                        # Store best detection so far
                        if BEST_DETECTION is None or box_area > BEST_DETECTION['area']:
                            annotated = results[0].plot()
                            center_x = img_width // 2
                            center_y = img_height // 2
                            roi_w = int(img_width * CENTER_THRESHOLD)
                            roi_h = int(img_height * CENTER_THRESHOLD)
                            cv2.rectangle(annotated, 
                                        (center_x - roi_w, center_y - roi_h),
                                        (center_x + roi_w, center_y + roi_h),
                                        (0, 255, 255), 2)
                            
                            BEST_DETECTION = {
                                'area': box_area,
                                'target_name': target_name,
                                'confidence': confidence,
                                'raw_img': raw_img.copy(),
                                'annotated_img': annotated.copy()
                            }
                            print(f"📈 New best detection: area={box_area:.0f} (tracking...)")
                        
                        # Check if size is decreasing (we've passed the closest point)
                        if len(OBSTACLE_SIZE_HISTORY) >= SIZE_DECREASE_THRESHOLD:
                            recent_sizes = list(OBSTACLE_SIZE_HISTORY)[-SIZE_DECREASE_THRESHOLD:]
                            is_decreasing = all(recent_sizes[i] > recent_sizes[i+1] 
                                              for i in range(len(recent_sizes)-1))
                            
                            if is_decreasing and BEST_DETECTION is not None:
                                print(f"\n🎯 SIZE PEAKED! Capturing best image (area={BEST_DETECTION['area']:.0f})")
                                
                                # Save the BEST detection
                                save_detection_image(
                                    ACTIVE_SNAP_ID, 
                                    BEST_DETECTION['target_name'],
                                    BEST_DETECTION['raw_img'],
                                    BEST_DETECTION['annotated_img']
                                )
                                
                                DETECTED_OBSTACLE_IDS.add(ACTIVE_SNAP_ID)
                                print(f"🔒 Obstacle {ACTIVE_SNAP_ID} marked as detected.")
                                
                                final_result_string = f"TARGET,{ACTIVE_SNAP_ID},{BEST_DETECTION['target_name']}"
                                s_log.sendall((final_result_string + "\n").encode('utf-8'))
                                print(f"✅ SENT TARGET RESULT: {final_result_string} (conf: {BEST_DETECTION['confidence']:.2f})")
                                
                                # Reset for next obstacle
                                last_sent_snap_id = ACTIVE_SNAP_ID
                                SNAP_ARMED = False
                                OBSTACLE_SIZE_HISTORY.clear()
                                BEST_DETECTION = None
                    else:
                        print(f"⚠️  Detections found but none are centered")
                else:
                    # No detection - might have moved away, save best if we have one
                    if BEST_DETECTION is not None and len(OBSTACLE_SIZE_HISTORY) >= 3:
                        print(f"\n⚠️  Lost detection. Saving best captured image...")
                        
                        save_detection_image(
                            ACTIVE_SNAP_ID, 
                            BEST_DETECTION['target_name'],
                            BEST_DETECTION['raw_img'],
                            BEST_DETECTION['annotated_img']
                        )
                        
                        DETECTED_OBSTACLE_IDS.add(ACTIVE_SNAP_ID)
                        
                        final_result_string = f"TARGET,{ACTIVE_SNAP_ID},{BEST_DETECTION['target_name']}"
                        s_log.sendall((final_result_string + "\n").encode('utf-8'))
                        print(f"✅ SENT TARGET RESULT: {final_result_string}")
                        
                        last_sent_snap_id = ACTIVE_SNAP_ID
                        SNAP_ARMED = False
                        OBSTACLE_SIZE_HISTORY.clear()
                        BEST_DETECTION = None

            # --- Display live inference ---
            annotated = results[0].plot()
            
            center_x = img_width // 2
            center_y = img_height // 2
            roi_w = int(img_width * CENTER_THRESHOLD)
            roi_h = int(img_height * CENTER_THRESHOLD)
            cv2.rectangle(annotated, 
                        (center_x - roi_w, center_y - roi_h),
                        (center_x + roi_w, center_y + roi_h),
                        (0, 255, 255), 2)
            
            # Show tracking status
            if SNAP_ARMED and BEST_DETECTION:
                status = f"Tracking... Best area: {BEST_DETECTION['area']:.0f}"
                cv2.putText(annotated, status, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            status_text = f"Detected: {len(DETECTED_OBSTACLE_IDS)} obstacles"
            cv2.putText(annotated, status_text, (10, img_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("YOLO Inference - Live", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except (ConnectionError, KeyboardInterrupt):
        print("Connection closed.")
    finally:
        s_stream.close()
        s_log.close()
        cv2.destroyAllWindows()
        print(f"\n📊 Final Stats: Detected {len(DETECTED_OBSTACLE_IDS)} unique obstacles")

if __name__ == "__main__":
    main()