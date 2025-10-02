import socket, struct
import numpy as np
import cv2
from ultralytics import YOLO
import threading
import time

# ---- CONFIG ----
PI_IP = "192.168.25.1"
PORT_STREAM = 5002  # RPi sends video stream HERE
PORT_LOG = 5001     # RPi listens for detection results HERE
MODEL = "runs/detect/aug_21/weights/best(125epochs).pt"

# ---- Load YOLO model ----
model = YOLO(MODEL)
# Map your custom class names here for easy reference
# Example: TARGET_CLASS_NAMES = {'3': 'circle', '4': 'triangle'}

# --- GLOBAL STATE ---
# This variable tracks the Obstacle ID (X from SNAPx) sent by the RPi
# It acts as a trigger and holds the ID required for the final output string.
ACTIVE_SNAP_ID = None 

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

# ---- NEW: RPi COMMAND LISTENER THREAD ----

def rpi_command_listener(log_sock):
    """
    Listens for a special command from the RPi signaling that a SNAP
    is happening and providing the Obstacle ID.
    """
    global ACTIVE_SNAP_ID
    print("RPi Command Listener started.")
    
    while True:
        try:
            # We assume the RPi sends the OBSTACLE_ID via the s_log socket 
            # (which is already open for log output). 
            # A simple protocol would be: "SNAP_ID,<ID>\n"
            data = log_sock.recv(1024).decode('utf-8').strip()
            
            if data.startswith("SNAP_ID,"):
                ACTIVE_SNAP_ID = data.split(',')[1].strip()
                print(f"\n*** SNAP DETECTED! Active Obstacle ID: {ACTIVE_SNAP_ID} ***")
            elif data:
                # Regular logs from RPi are still printed
                print(f"[RPi Log]: {data}")

        except Exception as e:
            print(f"RPi Command Listener Error: {e}")
            break

# ---- MAIN INFERENCE LOOP ----

def main():
    global ACTIVE_SNAP_ID
    
    # Setup socket for video stream (RPi sends to us)
    s_stream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to RPi video stream at {PI_IP}:{PORT_STREAM} ...")
    s_stream.connect((PI_IP, PORT_STREAM))

    # Setup a second socket for logging and commands (We send to RPi)
    s_log = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to RPi log channel at {PI_IP}:{PORT_LOG} ...")
    s_log.connect((PI_IP, PORT_LOG))
    
    print("All connections established. Starting inference...")

    # Start the command listener thread
    threading.Thread(target=rpi_command_listener, args=(s_log,)).start()
    
    # We use a placeholder for the detection result to prevent rapid re-sending
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

            # --- Run YOLO inference ---
            results = model.predict(source=img, imgsz=640, conf=0.25, verbose=False)
            
            # --- IMAGE DETECTION LOGIC ---
            
            # 1. Check if a SNAP event is active AND we haven't already processed this ID
            if ACTIVE_SNAP_ID and ACTIVE_SNAP_ID != last_sent_snap_id:
                
                # Assume detection is successful if ANY object is detected.
                # Otherwise, you would check if the confidence meets a threshold.
                if results and results[0].boxes:
                    # Get the top detected class name
                    top_cls_id = int(results[0].boxes.cls[0]) 
                    target_id = model.names[top_cls_id] # e.g., 'circle' or 'target'
                    
                    # Construct the final required string
                    final_result_string = f"TARGET,{ACTIVE_SNAP_ID},{target_id}"
                    
                    # Send result back to RPi
                    s_log.sendall((final_result_string + "\n").encode('utf-8')) 
                    print(f"\nSENT TARGET RESULT: {final_result_string}")
                    
                    # Mark this SNAP ID as processed
                    last_sent_snap_id = ACTIVE_SNAP_ID
                    
            # --- Display annotated image ---
            annotated = results[0].plot()
            cv2.imshow("YOLO Inference", annotated)
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