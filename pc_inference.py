import socket, struct
import numpy as np
import cv2
from ultralytics import YOLO
import threading
import time
import os
from datetime import datetime

# ---- CONFIG ----
PI_IP = "192.168.25.1"
PORT_STREAM = 5002
PORT_LOG = 5001
MODEL = "runs/detect/aug_21/weights/best(125epochs).pt"

# ---- IMAGE SAVE CONFIG ----
SAVE_DIR = "captured_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---- Load YOLO model ----
model = YOLO(MODEL)

# --- GLOBAL STATE ---
ACTIVE_SNAP_ID = None
SNAP_ARMED = False

# Store detected images for tiling
DETECTED_IMAGES = []  # List of tuples: (obstacle_id, target_name, annotated_image)

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

def save_detection_image(obstacle_id, target_name, raw_img, annotated_img):
    """
    Save both raw and annotated images locally with proper naming convention.
    Format: obs{ID}_target{NAME}_{timestamp}_raw.jpg
    This matches the requirement for displaying RAW images with bounding boxes.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save annotated RAW image with bounding box (THIS IS THE MAIN IMAGE FOR DISPLAY)
    # Using the format that matches the viewer expectations
    raw_annotated_filename = os.path.join(SAVE_DIR, f"obs{obstacle_id}_target{target_name}_{timestamp}_raw.jpg")
    cv2.imwrite(raw_annotated_filename, annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"✅ Saved RAW image with bounding box: {raw_annotated_filename}")
    
    # Optionally save the pure raw image without annotations (for reference)
    pure_raw_filename = os.path.join(SAVE_DIR, f"obs{obstacle_id}_target{target_name}_{timestamp}_pure.jpg")
    cv2.imwrite(pure_raw_filename, raw_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"📸 Saved pure raw image (no bbox): {pure_raw_filename}")
    
    # Add to global list for tiled display
    DETECTED_IMAGES.append((obstacle_id, target_name, annotated_img.copy()))
    
    # Update tiled display
    update_tiled_display()
    
    return raw_annotated_filename

def update_tiled_display():
    """
    Create a tiled display of all detected images in one window.
    This provides real-time visualization of all detections during the run.
    """
    if not DETECTED_IMAGES:
        return
    
    num_images = len(DETECTED_IMAGES)
    
    # Calculate grid dimensions (e.g., 3 columns max)
    cols = min(3, num_images)  # Max 3 images per row
    rows = (num_images + cols - 1) // cols
    
    # Resize all images to same size for tiling
    tile_size = (320, 240)  # Width x Height for each tile
    tiles = []
    
    for obstacle_id, target_name, img in DETECTED_IMAGES:
        # Resize image
        resized = cv2.resize(img, tile_size)
        
        # Add label with obstacle ID and target name
        label_bg = f"Obstacle: {obstacle_id}"
        label_target = f"Target: {target_name}"
        
        # Dark background for text readability
        cv2.rectangle(resized, (5, 5), (310, 60), (0, 0, 0), -1)
        
        # Add obstacle ID
        cv2.putText(resized, label_bg, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        # Add target name
        cv2.putText(resized, label_target, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        tiles.append(resized)
    
    # Pad with blank images if needed to fill grid
    while len(tiles) < rows * cols:
        blank = np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
        cv2.putText(blank, "Empty", (tile_size[0]//2 - 40, tile_size[1]//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        tiles.append(blank)
    
    # Create grid
    grid_rows = []
    for r in range(rows):
        row_tiles = tiles[r * cols:(r + 1) * cols]
        grid_row = np.hstack(row_tiles)
        grid_rows.append(grid_row)
    
    tiled_image = np.vstack(grid_rows)
    
    # Add header to tiled display
    header_height = 60
    header = np.zeros((header_height, tiled_image.shape[1], 3), dtype=np.uint8)
    header_text = f"DETECTED TARGETS - Total: {num_images}"
    cv2.putText(header, header_text, (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # Combine header and tiled images
    final_display = np.vstack([header, tiled_image])
    
    # Display tiled window
    cv2.imshow("Detected Targets - Tiled View", final_display)
    cv2.waitKey(1)

def save_final_tiled_image():
    """
    Save the final tiled image at the end of the run.
    This creates a single image with all detections for easy viewing.
    """
    if not DETECTED_IMAGES:
        print("⚠️ No images to tile.")
        return
    
    num_images = len(DETECTED_IMAGES)
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    tile_size = (640, 480)  # Larger tiles for final save
    tiles = []
    
    for obstacle_id, target_name, img in DETECTED_IMAGES:
        resized = cv2.resize(img, tile_size)
        
        # Add labels
        cv2.rectangle(resized, (10, 10), (630, 80), (0, 0, 0), -1)
        cv2.putText(resized, f"Obstacle: {obstacle_id}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)
        cv2.putText(resized, f"Target: {target_name}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
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
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(SAVE_DIR, f"tiled_results_{timestamp}.jpg")
    cv2.imwrite(output_filename, tiled_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\n✅ Saved final tiled image: {output_filename}")
    print(f"📊 Total detections: {num_images}")

# ---- RPi COMMAND LISTENER THREAD ----
def rpi_command_listener(log_sock):
    """
    Listen for commands from RPi:
    - SNAP_ID,{id}: Sets the active obstacle ID
    - SNAPREADY: Arms the detection system
    """
    global ACTIVE_SNAP_ID, SNAP_ARMED
    print("RPi Command Listener started.")
    
    while True:
        try:
            data = log_sock.recv(1024).decode('utf-8', errors='ignore').strip()
            if not data:
                continue

            for line in data.splitlines():
                line = line.strip()
                if not line:
                    continue

                if line.startswith("SNAP_ID,"):
                    ACTIVE_SNAP_ID = line.split(',', 1)[1].strip()
                    print(f"\n🎯 SNAP ID RECEIVED: Obstacle {ACTIVE_SNAP_ID}")
                    
                elif line == "SNAPREADY":
                    SNAP_ARMED = True
                    print(f"🔫 SNAP ARMED for Obstacle {ACTIVE_SNAP_ID}")
                    
                else:
                    # Log other messages from RPi
                    print(f"[RPi Log]: {line}")

        except Exception as e:
            print(f"❌ RPi Command Listener Error: {e}")
            break

# ---- MAIN INFERENCE LOOP ----
def main():
    global ACTIVE_SNAP_ID, SNAP_ARMED

    # Connect to RPi
    s_stream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to RPi video stream at {PI_IP}:{PORT_STREAM} ...")
    s_stream.connect((PI_IP, PORT_STREAM))
    print("✅ Video stream connected.")

    s_log = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to RPi log channel at {PI_IP}:{PORT_LOG} ...")
    s_log.connect((PI_IP, PORT_LOG))
    print("✅ Log channel connected.")

    print("\n" + "="*60)
    print("SYSTEM READY - Starting inference...")
    print("="*60 + "\n")

    # Start command listener thread
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

            # Keep a copy of raw image for saving
            raw_img = img.copy()

            # --- Run YOLO inference ---
            results = model.predict(source=img, imgsz=640, conf=0.25, verbose=False)

            # --- IMAGE DETECTION LOGIC ---
            if SNAP_ARMED and ACTIVE_SNAP_ID and ACTIVE_SNAP_ID != last_sent_snap_id:
                if results and results[0].boxes and len(results[0].boxes) > 0:
                    # Choose the highest-confidence detection
                    boxes = results[0].boxes
                    confs = boxes.conf.cpu().numpy()
                    best_idx = int(np.argmax(confs))
                    top_cls_id = int(boxes.cls[best_idx])
                    target_name = model.names[top_cls_id]
                    confidence = float(confs[best_idx])

                    # Get annotated image with bounding boxes
                    annotated = results[0].plot()

                    print(f"\n{'='*60}")
                    print(f"📸 DETECTION CAPTURED!")
                    print(f"   Obstacle ID: {ACTIVE_SNAP_ID}")
                    print(f"   Target: {target_name}")
                    print(f"   Confidence: {confidence:.2%}")
                    print(f"{'='*60}\n")

                    # Save images locally (RAW with bounding boxes)
                    saved_path = save_detection_image(ACTIVE_SNAP_ID, target_name, raw_img, annotated)

                    # Send result back to RPi
                    final_result_string = f"TARGET,{ACTIVE_SNAP_ID},{target_name}"
                    s_log.sendall((final_result_string + "\n").encode('utf-8'))
                    print(f"✅ SENT TO RPi: {final_result_string}\n")

                    # Update tracking
                    last_sent_snap_id = ACTIVE_SNAP_ID
                    SNAP_ARMED = False
                    
                else:
                    print(f"⚠️ SNAP ARMED but no objects detected for Obstacle {ACTIVE_SNAP_ID}")

            # --- Display live inference ---
            annotated = results[0].plot()
            
            # Add status indicator on live display
            status_text = f"ARMED - Obs {ACTIVE_SNAP_ID}" if SNAP_ARMED else "Monitoring"
            status_color = (0, 255, 0) if SNAP_ARMED else (200, 200, 200)
            
            cv2.rectangle(annotated, (5, 5), (300, 50), (0, 0, 0), -1)
            cv2.putText(annotated, status_text, (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Show detection count
            count_text = f"Detections: {len(DETECTED_IMAGES)}"
            cv2.putText(annotated, count_text, (10, 470), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("YOLO Inference - Live", annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("\n⚠️ ESC pressed - Exiting...")
                break
            elif key == ord('s') or key == ord('S'):  # 'S' key to save tiled image
                save_final_tiled_image()

    except (ConnectionError, KeyboardInterrupt) as e:
        print(f"\n⚠️ Connection closed: {type(e).__name__}")
    finally:
        print("\n" + "="*60)
        print("SHUTTING DOWN")
        print("="*60)
        
        # Save final tiled image before closing
        if DETECTED_IMAGES:
            save_final_tiled_image()
        
        s_stream.close()
        s_log.close()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Final Statistics:")
        print(f"   Total detections: {len(DETECTED_IMAGES)}")
        print(f"   Images saved in: {os.path.abspath(SAVE_DIR)}")
        print("\n✅ All resources closed. Goodbye!")

if __name__ == "__main__":
    main()