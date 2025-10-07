import serial
import time
import threading
import json
import re
import sys
import requests
from queue import Queue
from picamera import PiCamera
from picamera.array import PiRGBArray  # For camera streaming/snapshot capability
import socket, struct
import cv2

# --- CONFIGURATION ---
BT_PORT = '/dev/rfcomm0'
STM_PORT = '/dev/ttyACM0'  # Used for serial communication with STM
BAUDRATE = 115200
TASK_START_CMD = 'TASK1: START'
PC_ALGO_URL = 'http://192.168.25.14:5000/path'

# --- CAMERA/STREAM CONFIG ---
STREAM_PORT = 5002
LOG_PORT = 5001
CAMERA_RESOLUTION = (640, 480)
CAMERA_FRAMERATE = 24

# --- TIMER CONFIG ---
TIMEOUT_SECONDS = 360  # 6 minutes

# --- GLOBAL STATE & QUEUES ---
STATE = {
    'obstacles': {},
}
DIR_MAP = {'N': 0, 'E': 2, 'S': 4, 'W': 6}
REV_DIR_MAP = {0: 'N', 2: 'E', 4: 'S', 6: 'W'}

COMMAND_QUEUE = Queue()
PATH_QUEUE = Queue()

# --- VISION SYNC (NEW) ---
DETECT_EVENT = threading.Event()
DETECT_RESULT = {"ok": False, "obstacle_id": None, "target_id": None}

# --- TIMER ---
TIMER_START = None
TIMER_EXPIRED = threading.Event()

# --- Synchronization Primitives ---
CAMERA_READY = threading.Event()
STREAM_CONN = None
LOG_CONN = None 
CURRENT_OBSTACLE_ID = None

# --- SERIAL PORT SETUP ---
try:
    ser_bt = serial.Serial(port=BT_PORT, baudrate=BAUDRATE, timeout=0.01)
    print(f"Connected to Bluetooth on {BT_PORT}")
except serial.SerialException as e:
    print(f"ERROR: Bluetooth connection failed: {e}"); ser_bt = None
    
try:
    ser_stm = serial.Serial(port=STM_PORT, baudrate=BAUDRATE, timeout=0.05)
    time.sleep(2)
    print("Connected to STM:", ser_stm.portstr)
except serial.SerialException as e:
    print(f"ERROR: STM serial connection failed: {e}"); ser_stm = None

# -------------------------------------------------------------
# --- UTILITY FUNCTIONS ---

def get_obstacle_id_and_coords(match):
    obstacle_id = match.group(1).strip()
    x = int(match.group(2))
    y = int(match.group(3))
    return obstacle_id, x, y

def get_obstacle_id_and_dir(match):
    obstacle_id = match.group(1).strip()
    direction = match.group(2).strip()
    return obstacle_id, direction

def update_obstacle_state(command: str) -> bool:
    """
    Process a single complete command string:
      - ADD,<id>,(x,y)
      - FACE,<id>,[NESW]
      - DELETE,<id>

    Side effects:
      - Updates STATE['obstacles'] with +1 coord conversion for ADD.
      - Sends echo messages back to Android for FACE and DELETE as required.
    """
    global STATE
    
    if re.fullmatch(r'MAP-RESET', command.strip()):
        STATE['obstacles'].clear()
        with COMMAND_QUEUE.mutex: COMMAND_QUEUE.queue.clear()
        with PATH_QUEUE.mutex: PATH_QUEUE.queue.clear()
        print("[STATE] Cleared all obstacles from dictionary.")
        send_android_status("MAP RESET")     # status message
        return True

    # DELETE
    m = re.fullmatch(r'DELETE,(\w+)', command.strip())
    if m:
        ob_id = m.group(1).strip()
        if ob_id in STATE['obstacles']:
            del STATE['obstacles'][ob_id]
            print(f"[STATE] DELETED obstacle {ob_id}")
        else:
            print(f"[STATE] DELETE requested for unknown {ob_id}; ignoring dictionary removal")

        # Echo back to Android exactly as requested
        send_android_status(f"DELETE,{ob_id}")
        return True

    # ADD
    m = re.fullmatch(r'ADD,(\w+),\((\d+),(\d+)\)', command.strip())
    if m:
        ob_id = m.group(1).strip()
        x_0 = int(m.group(2))
        y_0 = int(m.group(3))

        # Android(0,0) -> Algo(1,1)
        x_1 = x_0 + 1
        y_1 = y_0 + 1

        STATE['obstacles'][ob_id] = STATE['obstacles'].get(ob_id, {'x': None, 'y': None, 'd': None, 'id': ob_id})
        STATE['obstacles'][ob_id]['x'] = x_1
        STATE['obstacles'][ob_id]['y'] = y_1

        print(f"[STATE] ADDED/UPDATED position of {ob_id} at ({x_1}, {y_1}) [Algo Coord]")
        # (Spec doesn't require echoing position line; we keep quiet here.)
        return True

    # FACE
    m = re.fullmatch(r'FACE,(\w+),([NESW])', command.strip())
    if m:
        ob_id = m.group(1).strip()
        direction_char = m.group(2).strip()

        if ob_id not in STATE['obstacles']:
            # If FACE arrives before ADD, create a shell entry so direction isn't lost
            STATE['obstacles'][ob_id] = {'x': None, 'y': None, 'd': None, 'id': ob_id}

        STATE['obstacles'][ob_id]['d'] = DIR_MAP.get(direction_char, 0)
        print(f"[STATE] UPDATED direction of {ob_id} to {direction_char} ({STATE['obstacles'][ob_id]['d']} deg)")

        # Echo exactly the FACE update back to Android (requirement)
        send_android_status(f"FACE,{ob_id},{direction_char}")
        return True

    return False

def format_state_to_json(state):
    valid_obstacles = []
    for ob_id, data in state['obstacles'].items():
        if data.get('x') is not None and data.get('y') is not None and data.get('d') is not None:
            valid_obstacles.append({
                'x': data['x'],
                'y': data['y'],
                'd': data['d'],
                'id': int(ob_id[1:])
            })
             
    # 2. Construct the final Python dictionary payload
    payload_dict = {
        "obstacles": valid_obstacles
    }
    
    # 3. Convert the Python dictionary to a JSON string
    return payload_dict
    
# -------------------------------------------------------------
#Andriod Helper Function 
def send_android_status(status_message):
    """Sends a formatted status update to the Android device via Bluetooth."""
    global ser_bt
    
    # 1. Format the required message string
    message = f"ROBOT-STATUS-UPDATE: {status_message}"
    
    if ser_bt and ser_bt.is_open:
        try:
            # 2. Append newline and encode the string to bytes
            data_to_send = (message + '\r\n').encode('utf-8')
            
            # 3. Write the data to the Bluetooth serial port
            ser_bt.write(data_to_send)
            print(f"[ANDROID OUT - STATUS] Sent: {message}")
           
        
            # 4. FORCE TRANSMISSION and wait briefly for the buffer to clear
            ser_bt.flush()
            time.sleep(0.05) # Small delay (50ms) to ensure write completes
            return True
        except Exception as e:
            print(f"ERROR: Failed to send status '{status_message}' to Android: {e}")
            return False
    else:
        print(f"WARN: Bluetooth serial port not open. Could not send status: {status_message}")
        return False
#  -------------------------------------------------------------------------------------------------
# --- STM helpers ---

def wait_for_stm_token(tokens, timeout=10.0):
    """
    Read lines until one of the tokens is found or timeout.
    Returns (matched_token, full_line) or (None, None) on timeout.
    """
    start = time.time()
    tokens = tokens if isinstance(tokens, (list, tuple, set)) else [tokens]
    while time.time() - start < timeout:
        try:
            line = ser_stm.readline().decode(errors='ignore').strip()
            if line:
                print(f"[STM IN] Received: {line}")
                for t in tokens:
                    if t in line:
                        return t, line
        except Exception as e:
            print(f"[STM READ ERROR] {e}")
        time.sleep(0.01)
    return None, None

def convert_instruction(instr):
    direction = instr[:2]
    valid_dir_turn = ["FL", "FR", "BR", "BL"]
    valid_dir_straight = ["FW", "BW"]
    valid_direction = valid_dir_turn + valid_dir_straight

    if direction not in valid_direction:
        return instr
    
    if direction in valid_dir_straight:
        distance = int(instr[2:])
        return f"{direction}{distance:03d}"

    elif direction in valid_dir_turn:
        angle = int(instr[2:])
        if angle == 0:
            angle = 90

        if direction == "FR":
            angle += 1
        elif direction == "FL":
            angle += -4
        elif direction == "BR":
            angle += -1
        elif direction == "BL":
            angle += -2

        return f"{direction}{angle:02d}"
# -------------------------------------------------------------
# --- ALGORITHM COMMUNICATION ---

def send_to_pc_algo(json_payload):
    """Sends JSON to PC Algo, processes response, and loads queues."""
    global COMMAND_QUEUE, PATH_QUEUE, TIMER_START
           
    if not json_payload['obstacles']:
        print("WARN: No complete obstacle definitions found. Skipping POST request.")
        send_android_status("No complete obstacle definitions found. Skipping POST request")
        return None

    try:
        print("\n--- Sending POST request to PC Algo ---")
        response = requests.post(PC_ALGO_URL, json=json_payload, timeout=10)
        response.raise_for_status()
        data = response.json().get('data', {})

        # --- PRINT RECEIVED JSON ---
        print("\n" + "="*20 + " RECEIVED JSON DATA " + "="*20)
        print(json.dumps(data, indent=4))
        print("="*60)

        # --- CLEAR EXISTING QUEUES ---
        with COMMAND_QUEUE.mutex: COMMAND_QUEUE.queue.clear()
        with PATH_QUEUE.mutex: PATH_QUEUE.queue.clear()

        # --- LOAD QUEUES ---
        commands = data.get('commands', [])
        path = data.get('path', [])

        for cmd in commands:
            COMMAND_QUEUE.put(cmd)

        for pos in path:
            PATH_QUEUE.put(pos)
            
        # START TIMER WHEN PATH IS FOUND
        TIMER_START = time.time()
        print(f"\n[TIMER] Started 6-minute countdown at {time.strftime('%H:%M:%S')}")
            
        send_android_status("PATH FOUND")
        print("\n\u2705 Queues Loaded Successfully.")
        print(f"  COMMAND_QUEUE size: {COMMAND_QUEUE.qsize()}")
        print(f"  PATH_QUEUE size: {PATH_QUEUE.qsize()}")

    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP Error communicating with PC Algo: {e}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to connect to PC Algo server: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during PC communication: {e}")

# -------------------------------------------------------------
# --- TIMER THREAD ---

def timer_thread():
    """Monitors 6-minute timer and sends 'X' to STM when expired."""
    global TIMER_START, ser_stm, TIMER_EXPIRED
    
    while True:
        if TIMER_START is not None and not TIMER_EXPIRED.is_set():
            elapsed = time.time() - TIMER_START
            remaining = TIMEOUT_SECONDS - elapsed
            
            if remaining <= 0:
                print("\n" + "="*40)
                print("[TIMER] 6 MINUTES EXPIRED! Sending 'X' to STM.")
                print("="*40)
                
                if ser_stm and ser_stm.is_open:
                    try:
                        ser_stm.write(b"X\n")
                        print("[STM OUT] Sent: X (TIMEOUT)")
                        send_android_status("TIMEOUT: 6 minutes expired")
                    except Exception as e:
                        print(f"ERROR sending 'X' to STM: {e}")
                
                TIMER_EXPIRED.set()
                TIMER_START = None  # Reset timer
            
        time.sleep(0.5)  # Check every 0.5 seconds

# -------------------------------------------------------------
# --- IMAGE DETECTION LISTENER THREAD ---

def image_detection_listener_thread():
    """Listens for TARGET,<obstacle_id>,<target_id> from PC and signals DETECT_EVENT."""
    global LOG_CONN, ser_bt, CURRENT_OBSTACLE_ID, DETECT_EVENT, DETECT_RESULT

    if not LOG_CONN:
        print("IMAGE LISTENER: Log connection not established.")
        return

    while True:
        try:
            data = LOG_CONN.recv(1024).decode('utf-8', errors='ignore').strip()
            if not data:
                continue

            # Multiple lines can arrive at once; handle each
            for line in data.splitlines():
                line = line.strip()
                if not line:
                    continue

                if line.startswith("TARGET,"):
                    parts = line.split(',')
                    if len(parts) == 3:
                        obstacle_id = parts[1].strip()
                        target_id   = parts[2].strip()

                        if obstacle_id != str(CURRENT_OBSTACLE_ID):
                            print(f"WARN: Target ID mismatch. Expected {CURRENT_OBSTACLE_ID}, got {obstacle_id}.")

                        # Record result and wake execution thread
                        DETECT_RESULT.update({"ok": True, "obstacle_id": obstacle_id, "target_id": target_id})
                        DETECT_EVENT.set()

                        # Forward to Android
                        final_target_string = f"TARGET,{obstacle_id},{target_id}"
                        send_android_status(f"TARGET {obstacle_id} -> {target_id}")
                        print(f"\n[ANDROID OUT - TARGET] Sending: {final_target_string}")
                        if ser_bt and ser_bt.is_open:
                            ser_bt.write((final_target_string + '\n').encode('utf-8'))

                        CURRENT_OBSTACLE_ID = None
                        continue

                # Otherwise just log
                print(f"[PC LOG] {line}")

        except Exception as e:
            print(f"IMAGE LISTENER ERROR: {e}")
            # Wake the exec thread so it doesn't hang forever
            DETECT_RESULT.update({"ok": False, "obstacle_id": str(CURRENT_OBSTACLE_ID), "target_id": None})
            DETECT_EVENT.set()
            break

# -------------------------------------------------------------
# --- CAMERA STREAMING THREAD ---

def camera_stream_thread(conn_stream):
    """Initializes camera and streams frames to the PC for image detection."""
    global CAMERA_READY

    try:
        camera = PiCamera()
        camera.resolution = CAMERA_RESOLUTION
        camera.framerate = CAMERA_FRAMERATE
        raw = PiRGBArray(camera, size=CAMERA_RESOLUTION)
        time.sleep(2)  # Camera warm-up

        CAMERA_READY.set()  # Signal that the camera is ready
        print("\n\u2705 Camera initialized and stream ready.")

        for frame in camera.capture_continuous(raw, format="bgr", use_video_port=True):
            img = frame.array
            # JPEG encode to shrink size
            ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                raw.truncate(0)
                continue
            data = buf.tobytes()

            # Send 8-byte length header + payload
            conn_stream.sendall(struct.pack('>Q', len(data)) + data)
            raw.truncate(0)

    except Exception as e:
        print(f"CAMERA STREAM ERROR: {e}")
    finally:
        if 'camera' in locals():
            camera.close()
        print("Camera streaming stopped.")

# -------------------------------------------------------------
# --- EXECUTION THREAD (MOTION CONTROL LOOP) ---

def execution_thread():
    """
    Manages command dequeue, STM communication, and SNAP logic.

    Updated behavior:
    1) Send cmd -> wait for 'ACK' -> send ROBOT pose to Android.
    2) Then wait for 'END' -> only then fetch & send next command.
    3) For 'SNAP_#' or 'SNAP#': send 'SNAP' to STM -> wait 'SNAPREADY' -> trigger PC detection -> continue.
    4) Send movement status to Android based on command: FW/BW/FR/FL.
    5) New: send status 'SNAPPING' when SNAP sent, 'IMAGE DETECTION' on SNAPREADY.
    6) New: after ACK on regular move, also send 'MOVE,<normalized_cmd>' to Android (e.g., MOVE,FW100).
    """
    
    global ser_stm, ser_bt, CURRENT_OBSTACLE_ID

    if not ser_stm or not ser_stm.is_open:
        print("EXECUTION THREAD: STM serial not active.")
        return

    while True:
        try:
            # Check if timer expired
            if TIMER_EXPIRED.is_set():
                print("[EXECUTION] Timer expired. Stopping execution.")
                break
                
            if COMMAND_QUEUE.empty():
                time.sleep(0.1)
                continue

            current_command = COMMAND_QUEUE.get_nowait()

            if current_command == "FIN":
                print("\nEXECUTION: FIN command received. Path complete.")
                send_android_status("PATH COMPLETE")
                continue 
            
            # --- inside execution_thread(), replace your SNAP handling with this ---
            # Accept only SNAP<digits>_<letter>, e.g., SNAP1_C
            snap_m = re.match(r'^SNAP(?P<id>\d+)_([A-Za-z])$', current_command.strip(), flags=re.IGNORECASE)
            if snap_m:
                obstacle_id = snap_m.group('id')
                CURRENT_OBSTACLE_ID = obstacle_id
                print(f"\nEXECUTION: SNAP command ({current_command}) received. Obstacle ID: {obstacle_id}.")

                # Send only 'SNAP' to STM
                ser_stm.write(b"SNAP\n")
                print("[STM OUT] Sent: SNAP")
                send_android_status("SNAPPING")

                # 1) Wait for STM SNAPREADY
                token, _ = wait_for_stm_token("SNAPREADY", timeout=10.0)
                if not token:
                    print("EXECUTION ERROR: SNAPREADY timeout.")
                    send_android_status("IMAGE DETECTION ERROR")
                    continue

                send_android_status("IMAGE DETECTION")
                time.sleep(8)
                
                # 2) INFORM PC: first 'SNAPREADY' (to arm), then which obstacle this is
                if LOG_CONN:
                    try:
                        LOG_CONN.sendall(b"SNAPREADY\n")   # <-- PC arms on this
                        snap_id_msg = f"SNAP_ID,{obstacle_id}\n"
                        LOG_CONN.sendall(snap_id_msg.encode('utf-8'))
                        print(f"[PC OUT - SNAP TRIGGER] Sent: SNAPREADY + {snap_id_msg.strip()}")
                    except Exception as e:
                        print(f"ERROR sending SNAP notifications to PC: {e}")

                # 3) Wait up to 8s for TARGET from image listener
                DETECT_EVENT.clear()
                DETECT_RESULT.update({"ok": False, "obstacle_id": None, "target_id": None})

                if not DETECT_EVENT.wait(timeout=8.0):
                    msg = f"VISION TIMEOUT: no target detected for obstacle {obstacle_id}"
                    print(msg)
                    send_android_status(msg)
                    # proceed to next command
                    continue

                # If signaled: proceed regardless of success/failure flag (PC only sends success in your setup)
                if not DETECT_RESULT.get("ok", False):
                    # Listener hit an exception; notify and continue
                    print("VISION FAIL: listener error or no result.")
                    send_android_status("VISION FAIL: no result")
                    continue

                print(f"VISION OK: obstacle {DETECT_RESULT['obstacle_id']} -> target {DETECT_RESULT['target_id']}")
                # Move on to next queued command
                continue

             # --- Regular movement command ---
            
            # Normalize instruction for STM (padding/angle tweaks)
            normalized_cmd = convert_instruction(current_command)
            print(f"\nEXECUTION: Sending command to STM: {normalized_cmd}")
            ser_stm.write((normalized_cmd + '\n').encode('utf-8'))
            print(f"[STM OUT] Sent: {normalized_cmd}")

            # ---- Send movement status to Android based on command prefix ----
            prefix = normalized_cmd[:2]
            if   prefix == "FW":
                send_android_status("Moving North")
            elif prefix == "BW":
                send_android_status("Moving South")
            elif prefix == "FR":
                send_android_status("Turning East")
            elif prefix == "FL":
                send_android_status("Turning North")
            # (extend for BR/BL if needed)

            # 1) WAIT FOR ACK (per requirement #2)
            ack_token, ack_line = wait_for_stm_token("ACK", timeout=5.0)
            if not ack_token:
                print("EXECUTION ERROR: ACK timeout. Command not confirmed.")
                continue

            # 2) After ACK, send ROBOT position to Android from PATH_QUEUE
            if not PATH_QUEUE.empty():
                next_pos = PATH_QUEUE.get_nowait()
                # --- COORDINATE CONVERSION: Algo(1,1)->Android(0,0) ---
                x_0 = (next_pos.get('x', 1) - 1)
                y_0 = (next_pos.get('y', 1) - 1)
                direction_char = REV_DIR_MAP.get(next_pos.get('d', 0), 'U')
                robot_string = f"ROBOT,{x_0},{y_0},{direction_char}"
                
                #Sending ROBOT (x,y)
                if ser_bt and ser_bt.is_open:
                    ser_bt.write((robot_string + '\n').encode('utf-8'))
                print(f"[ANDROID OUT] Sent: {robot_string}")
                send_android_status(robot_string)
                
                #Sending Movement data
                if ser_bt and ser_bt.is_open:
                    move_line = f"MOVE,{normalized_cmd}"
                    ser_bt.write((move_line + '\n').encode('utf-8'))
                    print(f"[ANDROID OUT] Sent: {move_line}")
                    send_android_status(move_line)
            else:
                print("WARN: PATH_QUEUE empty at ACK; no robot pose to send.")
                send_android_status("WARN: PATH_QUEUE empty")

            # 3) WAIT FOR END (per requirement #3) BEFORE moving to next command
            end_token, end_line = wait_for_stm_token("END", timeout=30.0)
            if not end_token:
                print("EXECUTION ERROR: END timeout. Movement may not have completed.")
                # You can decide to continue or requeue current_command if needed.
                continue

            # END received -> loop to next command
            # (Nothing to do here; the while loop will fetch the next command.)

        except Exception as e:
            print(f"EXECUTION THREAD ERROR: {e}")
            break

    print("Execution thread finished.")

# -------------------------------------------------------------
# --- BLUETOOTH HANDLER (INPUT THREAD) ---

def handle_bt_commands():
    """Robustly parse BT input stream for ADD/DELETE/FACE/MAP_RESET and TASK_START."""
    global ser_bt
    if not ser_bt or not ser_bt.is_open:
        print("BLUETOOTH HANDLER: Serial not active.")
        return

    # Add MAP_RESET to the scanner. Accept underscore or hyphen.
    CMD_RE = re.compile(
        r'(?:ADD,\w+,\(\d+,\d+\))|'
        r'(?:DELETE,\w+)|'
        r'(?:FACE,\w+,[NESW])|'
        r'(?:MAP-RESET)'
    )

    remainder = ""

    def _status_echo(cmd_str: str):
        send_android_status(f"Received {cmd_str}")

    while True:
        try:
            incoming = ser_bt.read(ser_bt.in_waiting or 1)
            if not incoming:
                time.sleep(0.01)
                continue

            chunk = incoming.decode('utf-8', errors='ignore')
            buffer = remainder + chunk

            # 1) Extract all complete commands
            matches = list(CMD_RE.finditer(buffer))
            last_end = 0
            for m in matches:
                cmd_str = m.group(0)
                _status_echo(cmd_str)
                if not update_obstacle_state(cmd_str):
                    print(f"[PARSE WARN] Matched but not processed: {cmd_str}")
                last_end = m.end()

            # 2) Remove all matched command bytes
            buffer_after_cmds = buffer[last_end:]

            # 3) Handle TASK_START flags anywhere in the leftover buffer
            #    Trigger once per occurrence; remove them so they won't retrigger.
            triggered = False
            while TASK_START_CMD in buffer_after_cmds:
                triggered = True
                i = buffer_after_cmds.find(TASK_START_CMD)
                buffer_after_cmds = buffer_after_cmds[:i] + buffer_after_cmds[i+len(TASK_START_CMD):]

            if triggered:
                print("\n" + "="*40)
                print(f"CMD SIGNAL: '{TASK_START_CMD}' received. Processing state for Algo...")
                final_json_payload = format_state_to_json(STATE)
                send_android_status("Starting Pathfinding Algorithm")
                print(json.dumps(final_json_payload))
                send_to_pc_algo(final_json_payload)

            # 4) Whatever remains could be an incomplete tail; keep it.
            remainder = buffer_after_cmds

        except Exception as e:
            print(f"An error occurred in BT handler: {e}")
            break


# -------------------------------------------------------------
# --- MAIN EXECUTION ---

if __name__ == "__main__":
    
    if not ser_bt: 
        print("No BT")
    else:
        # --- TCP SERVER SETUP ---
        # 1. Setup server sockets (for image streaming and log receiving)
        srv_stream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv_stream.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv_stream.bind(('0.0.0.0', STREAM_PORT))
        srv_stream.listen(1)

        srv_log = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv_log.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv_log.bind(('0.0.0.0', LOG_PORT))
        srv_log.listen(1)

        # 2. Wait for PC connection (blocking calls)
        print(f"Waiting for PC Image Stream connection on {STREAM_PORT} ...")
        STREAM_CONN, _ = srv_stream.accept()
        print("\u2705 PC Image Stream connected.")

        print(f"Waiting for PC Log connection on {LOG_PORT} ...")
        LOG_CONN, _ = srv_log.accept()
        print("\u2705 PC Log connected.")

        # --- THREAD START ---
        
        #1. Start Bluetooth/Input Handler
        bt_thread = threading.Thread(target=handle_bt_commands)
        bt_thread.daemon = True
        bt_thread.start()
        
        #2. Start Camera Stream
        stream_thread = threading.Thread(target=camera_stream_thread, args=(STREAM_CONN,))
        stream_thread.daemon = True
        stream_thread.start()
        
        # 3. Start Image Detection Listener (Always running to catch results)
        log_listener_thread = threading.Thread(target=image_detection_listener_thread)
        log_listener_thread.daemon = True
        log_listener_thread.start()
           
        # 4. Start Execution/Motion Control
        exec_thread = threading.Thread(target=execution_thread)
        exec_thread.daemon = True
        exec_thread.start()
        
        # 5. Start Timer Thread
        timer_monitor = threading.Thread(target=timer_thread)
        timer_monitor.daemon = True
        timer_monitor.start()
        
        CAMERA_READY.wait(timeout=5)

        print("System initialization complete. Waiting for user input (Bluetooth/Console)...")
        send_android_status("ROBOT READY")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting program.")
            send_android_status("EXITING PROGRAM")
        finally:
            # Cleanup connections
            if ser_bt and ser_bt.is_open: ser_bt.close()
            # if ser_stm and ser_stm.is_open: ser_stm.close()
            srv_stream.close()
            srv_log.close()
            print("All connections and resources closed.")