import socket, struct, time, threading
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import serial

# --- Serial Port Setup for STM communication ---
try:
    ser = serial.Serial(
        port='/dev/ttyACM0',
        baudrate=115200,
        timeout=1
    )
    time.sleep(2)
    print("Serial port connected to:", ser.portstr)
except serial.SerialException as e:
    print(f"Error connecting to serial port: {e}")
    ser = None

# --- Counters for detection logic ---
non_marker_count = 0
marker_count = 0
NON_MARKER_THRESHOLD = 10
MARKER_THRESHOLD = 10

# --- TCP Server Setup ---
HOST, PORT_STREAM = '0.0.0.0', 5002
HOST, PORT_LOG = '0.0.0.0', 5001

srv_stream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv_stream.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv_stream.bind((HOST, PORT_STREAM))
srv_stream.listen(1)

srv_log = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv_log.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv_log.bind((HOST, PORT_LOG))
srv_log.listen(1)

def handle_pc_logs(conn):
    """Listens for log messages and commands from the PC."""
    global non_marker_count, marker_count

    while True:
        try:
            data = conn.recv(1024).decode('utf-8')
            if not data:
                break

            # Check for marker detection
            if "MARKER_DETECTED" in data:
                marker_count += 1
               # print(f"Marker detection #{marker_count}")

                # Send INVALID every 10 marker detections
                if marker_count >= MARKER_THRESHOLD:
                   # print(f"Marker threshold reached ({MARKER_THRESHOLD}). Sending 'INVALID' to STM.")
                    if ser and ser.is_open:
                        invalid_command = "INVALID\n"
                        ser.write(invalid_command.encode('utf-8'))
                        print("Sent:", invalid_command.strip())
                    marker_count = 0  # Reset counter

            # Handle other detections (non-marker)
            elif "Detected items:" in data and "marker" not in data.lower():
                non_marker_count += 1
                print(f"Non-marker detection #{non_marker_count}")

                # Send VALID every 10 non-marker detections
                if non_marker_count >= NON_MARKER_THRESHOLD:
                    print(f"Non-marker threshold reached ({NON_MARKER_THRESHOLD}). Sending 'VALID' to STM.")
                    if ser and ser.is_open:
                        valid_command = "VALID\n"
                        ser.write(valid_command.encode('utf-8'))
                        print("Sent:", valid_command.strip())
                    non_marker_count = 0  # Reset counter

            # Print all log data to RPi console
           # print("PC Log:", data.strip())

        except (ConnectionResetError, BrokenPipeError):
            print("PC log connection closed.")
            break
        except Exception as e:
            print(f"An error occurred in log handler: {e}")
            break
# --- Main execution ---
print(f"Waiting for PC connection on {PORT_STREAM} ...")
conn_stream, addr_stream = srv_stream.accept()
print("PC stream connected:", addr_stream)

print(f"Waiting for PC log connection on {PORT_LOG} ...")
conn_log, addr_log = srv_log.accept()
print("PC log connected:", addr_log)

# Start the log listener thread
log_thread = threading.Thread(target=handle_pc_logs, args=(conn_log,))
log_thread.daemon = True
log_thread.start()

try:
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 24
    raw = PiRGBArray(camera, size=(640, 480))
    time.sleep(2)

    for frame in camera.capture_continuous(raw, format="bgr", use_video_port=True):
        img = frame.array
        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raw.truncate(0)
            continue
        data = buf.tobytes()
        conn_stream.sendall(struct.pack('>Q', len(data)) + data)
        raw.truncate(0)

except (BrokenPipeError, ConnectionResetError, KeyboardInterrupt):
    print("Stream stopped.")
finally:
    try: conn_stream.close()
    except: pass
    try: conn_log.close()
    except: pass
    srv_stream.close()
    srv_log.close()
    camera.close()
    if ser and ser.is_open:
        ser.close()
    print("All connections closed.")
