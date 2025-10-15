import socket
import struct
import numpy as np
import cv2
from ultralytics import YOLO
import os
from datetime import datetime

# ---- CONFIG ----
PI_IP = "192.168.25.1"  # Change to your RPi IP
PORT_STREAM = 5002
PORT_LOG = 5001
MODEL = "runs/detect/aug_21/weights/best(125epochs).pt"  # Change to your model path

# ---- IMAGE SAVE CONFIG ----
SAVE_DIR = "test_captures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---- IMAGE ID MAPPING ----
IMAGE_ID_MAP = {
    '11': {'name': 'Number 1', 'id': '11'},
    '12': {'name': 'Number 2', 'id': '12'},
    '13': {'name': 'Number 3', 'id': '13'},
    '14': {'name': 'Number 4', 'id': '14'},
    '15': {'name': 'Number 5', 'id': '15'},
    '16': {'name': 'Number 6', 'id': '16'},
    '17': {'name': 'Number 7', 'id': '17'},
    '18': {'name': 'Number 8', 'id': '18'},
    '19': {'name': 'Number 9', 'id': '19'},
    '20': {'name': 'Alphabet A', 'id': '20'},
    '21': {'name': 'Alphabet B', 'id': '21'},
    '22': {'name': 'Alphabet C', 'id': '22'},
    '23': {'name': 'Alphabet D', 'id': '23'},
    '24': {'name': 'Alphabet E', 'id': '24'},
    '25': {'name': 'Alphabet F', 'id': '25'},
    '26': {'name': 'Alphabet G', 'id': '26'},
    '27': {'name': 'Alphabet H', 'id': '27'},
    '28': {'name': 'Alphabet S', 'id': '28'},
    '29': {'name': 'Alphabet T', 'id': '29'},
    '30': {'name': 'Alphabet U', 'id': '30'},
    '31': {'name': 'Alphabet V', 'id': '31'},
    '32': {'name': 'Alphabet W', 'id': '32'},
    '33': {'name': 'Alphabet X', 'id': '33'},
    '34': {'name': 'Alphabet Y', 'id': '34'},
    '35': {'name': 'Alphabet Z', 'id': '35'},
    '36': {'name': 'Up Arrow', 'id': '36'},
    '37': {'name': 'Down Arrow', 'id': '37'},
    '38': {'name': 'Left Arrow', 'id': '38'},
    '39': {'name': 'Right Arrow', 'id': '39'},
    '40': {'name': 'Circle', 'id': '40'},
}

def get_display_info(model_class_name):
    """Get display name and image ID from model class name."""
    info = IMAGE_ID_MAP.get(str(model_class_name), {'name': f'Unknown ({model_class_name})', 'id': model_class_name})
    return info['name'], info['id']

def recv_exact(sock, n):
    """Receive exactly n bytes or raise an error if connection closes."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf.extend(chunk)
    return bytes(buf)

# ---- Load YOLO model ----
print("Loading YOLO model...")
model = YOLO(MODEL)
print("✅ Model loaded successfully.")

# ---- Connect to RPi ----
s_stream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(f"Connecting to RPi video stream at {PI_IP}:{PORT_STREAM} ...")
s_stream.connect((PI_IP, PORT_STREAM))
print("✅ Video stream connected.")

s_log = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(f"Connecting to RPi log channel at {PI_IP}:{PORT_LOG} ...")
s_log.connect((PI_IP, PORT_LOG))
print("✅ Log channel connected.")

print("\n" + "="*60)
print("CAMERA & DETECTION TEST - Live Inference")
print("Controls:")
print("  - Press 'S' to save current frame")
print("  - Press 'ESC' to exit")
print("="*60 + "\n")

frame_count = 0
saved_count = 0

try:
    while True:
        # --- Read video stream frame ---
        hdr = recv_exact(s_stream, 8)
        (length,) = struct.unpack(">Q", hdr)
        payload = recv_exact(s_stream, length)
        img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            continue
        
        frame_count += 1
        
        # --- Run YOLO inference ---
        results = model.predict(source=img, imgsz=640, conf=0.25, verbose=False)
        
        # --- Get annotated image ---
        annotated = results[0].plot()
        
        # --- Display detection info ---
        if results and results[0].boxes and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            classes = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            
            # Count detections (excluding markers)
            non_marker_count = 0
            for i, cls_id in enumerate(classes):
                class_name = model.names[int(cls_id)]
                if class_name.lower() != "marker":
                    non_marker_count += 1
                    display_name, image_id = get_display_info(class_name)
                    conf = confs[i]
                    
                    # Print detection info
                    if frame_count % 30 == 0:  # Print every 30 frames to avoid spam
                        print(f"Detected: {display_name} (ID: {image_id}) - Confidence: {conf:.2%}")
            
            # Add detection count to display
            info_text = f"Detections: {non_marker_count} | Frames: {frame_count} | Saved: {saved_count}"
        else:
            info_text = f"No detections | Frames: {frame_count} | Saved: {saved_count}"
        
        # --- Add info overlay ---
        cv2.rectangle(annotated, (5, 5), (550, 50), (0, 0, 0), -1)
        cv2.putText(annotated, info_text, (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # --- Show live feed ---
        cv2.imshow("Camera & Detection Test - Live Feed", annotated)
        
        # --- Handle keyboard input ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\n⚠️ ESC pressed - Exiting...")
            break
        elif key == ord('s') or key == ord('S'):  # Save frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SAVE_DIR, f"test_frame_{timestamp}.jpg")
            cv2.imwrite(filename, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
            print(f"📸 Saved frame: {filename}")

except (ConnectionError, KeyboardInterrupt) as e:
    print(f"\n⚠️ Connection closed: {type(e).__name__}")
finally:
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print(f"Total frames processed: {frame_count}")
    print(f"Total frames saved: {saved_count}")
    print(f"Images saved in: {os.path.abspath(SAVE_DIR)}")
    print("="*60)
    
    s_stream.close()
    s_log.close()
    cv2.destroyAllWindows()
    print("\n✅ All resources closed. Goodbye!")