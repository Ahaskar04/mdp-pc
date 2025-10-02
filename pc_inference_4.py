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
# Example (optional): TARGET_CLASS_NAMES = {'3': 'circle', '4': 'triangle'}

# --- GLOBAL STATE ---
ACTIVE_SNAP_ID = None   # set by SNAP_ID,<id>
SNAP_ARMED = False      # set True by SNAPREADY (from RPi after STM is ready)

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

# ---- RPi COMMAND LISTENER THREAD ----
def rpi_command_listener(log_sock):
    """
    Listens for:
      - SNAP_ID,<ID>   -> remember the obstacle id
      - SNAPREADY      -> arm detection (start looking for the target)
      - other lines    -> just print
    """
    global ACTIVE_SNAP_ID, SNAP_ARMED
    print("RPi Command Listener started.")
    while True:
        try:
            data = log_sock.recv(1024).decode('utf-8').strip()
            if not data:
                continue

            # Support multiple lines in one read
            for line in data.splitlines():
                line = line.strip()
                if not line:
                    continue

                if line.startswith("SNAP_ID,"):
                    ACTIVE_SNAP_ID = line.split(',', 1)[1].strip()
                    print(f"\n*** SNAP ID RECEIVED. Active Obstacle ID: {ACTIVE_SNAP_ID} ***")
                    # Do NOT arm here. We only arm on explicit SNAPREADY.
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
    threading.Thread(target=rpi_command_listener, args=(s_log,), daemon=True).start()

    # Prevent duplicate sends per SNAP cycle
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
            # Only allow sending TARGET after SNAPREADY (SNAP_ARMED == True)
            if SNAP_ARMED and ACTIVE_SNAP_ID and ACTIVE_SNAP_ID != last_sent_snap_id:
                if results and results[0].boxes and len(results[0].boxes) > 0:
                    # Choose the highest-confidence detection explicitly
                    boxes = results[0].boxes
                    confs = boxes.conf.cpu().numpy()
                    best_idx = int(np.argmax(confs))
                    top_cls_id = int(boxes.cls[best_idx])
                    target_name = model.names[top_cls_id]  # string class name

                    final_result_string = f"TARGET,{ACTIVE_SNAP_ID},{target_name}"

                    # Send result back to RPi
                    s_log.sendall((final_result_string + "\n").encode('utf-8'))
                    print(f"\nSENT TARGET RESULT: {final_result_string}")

                    # Update cycle state: prevent duplicates until a new SNAP_ID comes
                    last_sent_snap_id = ACTIVE_SNAP_ID
                    SNAP_ARMED = False  # de-arm until the next SNAPREADY
                    # (You can also clear ACTIVE_SNAP_ID here if your RPi expects that.)
                    # ACTIVE_SNAP_ID = None

            # --- Display annotated image (optional) ---
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
