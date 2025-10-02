import socket, struct
import numpy as np
import cv2
from ultralytics import YOLO

# ---- CONFIG ----
PI_IP   = "192.168.25.1"      
PORT    = 5000
MODEL   = "runs/detect/aug_21/weights/best(125epochs).pt"  

# ---- Load YOLO model ----
model = YOLO(MODEL)

def recv_exact(sock, n):
    """Receive exactly n bytes or raise an error if connection closes."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf.extend(chunk)
    return bytes(buf)

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to RPi at {PI_IP}:{PORT} ...")
    s.connect((PI_IP, PORT))
    print("Connected. Starting inference...")

    try:
        while True:
            # --- Read 8-byte length header ---
            hdr = recv_exact(s, 8)
            (length,) = struct.unpack(">Q", hdr)

            # --- Read JPEG payload ---
            payload = recv_exact(s, length)

            # --- Decode JPEG to BGR image ---
            img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            # --- Run YOLO inference ---
            results = model.predict(source=img, imgsz=640, conf=0.25, verbose=False)
            annotated = results[0].plot()  # draws boxes/labels

            # --- Display ---
            cv2.imshow("YOLO Inference", annotated)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

    except (ConnectionError, KeyboardInterrupt):
        print("Connection closed.")
    finally:
        s.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()