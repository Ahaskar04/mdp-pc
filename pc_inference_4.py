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

# Create directory for saved images
SAVE_DIR = "detected_images"
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
    
    # Calculate grid dimensions (e.g., 2 columns)
    cols = min(3, num_images)  # Max 3 images per row
    rows = (num_images + cols - 1) // cols
    
    # Resize all images to same size for tiling
    tile_size = (320, 240)  # Width x Height for each tile
    tiles = []
    
    for obstacle_id, target_name, img in DETECTED_IMAGES:
        # Resize image
        resized = cv2.resize(img, tile_size)
        
        # Add label with obstacle ID and target name
        label = f"Obs {obstacle_id}: {target_name}"
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
    global ACTIVE_SNAP_ID, SNAP_ARMED

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

                    # Get annotated image with bounding boxes
                    annotated = results[0].plot()

                    # ✨ NEW: Save images locally
                    save_detection_image(ACTIVE_SNAP_ID, target_name, raw_img, annotated)

                    final_result_string = f"TARGET,{ACTIVE_SNAP_ID},{target_name}"

                    # Send result back to RPi
                    s_log.sendall((final_result_string + "\n").encode('utf-8'))
                    print(f"\nSENT TARGET RESULT: {final_result_string}")

                    last_sent_snap_id = ACTIVE_SNAP_ID
                    SNAP_ARMED = False

            # --- Display live inference (optional) ---
            annotated = results[0].plot()
            cv2.imshow("YOLO Inference - Live", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except (ConnectionError, KeyboardInterrupt):
        print("Connection closed.")
    finally:
        s_stream.close()
        s_log.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()