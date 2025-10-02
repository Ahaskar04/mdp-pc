import socket, struct
import numpy as np
import cv2
from ultralytics import YOLO

# ---- CONFIG ----
PI_IP   = "192.168.25.1"
PORT_STREAM = 5002  # Port for receiving video stream
PORT_LOG = 5001     # Port for sending logs/commands
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
    # Setup socket for video stream
    s_stream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to RPi video stream at {PI_IP}:{PORT_STREAM} ...")
    s_stream.connect((PI_IP, PORT_STREAM))

    # Setup a second socket for logging and commands
    s_log = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to RPi log channel at {PI_IP}:{PORT_LOG} ...")
    s_log.connect((PI_IP, PORT_LOG))
    
    print("All connections established. Starting inference...")

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

            # --- Process results and log to RPi ---
            detected_names = []
            for r in results:
                for c in r.boxes.cls:
                    # Use the model's names dictionary to get the string label
                    class_name = model.names[int(c)]
                    detected_names.append(class_name)
            
            if detected_names:
                # Get unique names and sort them for clean logging
                unique_names = sorted(list(set(detected_names)))
                log_message = f"Detected items: {unique_names}"
                print(log_message) # Log to local PC console
                s_log.sendall((log_message + "\n").encode('utf-8')) # Send to RPi

                # Check for the marker name
                if "marker" in unique_names:
                    s_log.sendall("MARKER_DETECTED\n".encode('utf-8')) # Send command to RPI

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