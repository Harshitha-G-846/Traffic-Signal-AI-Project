import cv2
from ultralytics import YOLO
import serial
import time
import csv
import os
from datetime import datetime

# ==========================================================
# SETTINGS
# ==========================================================
ARDUINO_PORT = 'COM5'          # Change to your Arduino COM port
BAUD_RATE = 9600
VIDEO_SOURCE = 'traffic_video4.mp4'   # Use 0 for webcam
MODEL_NAME = 'yolov8s.pt'
CONFIDENCE_THRESHOLD = 0.35

# Divider position (adjust to align with actual road separator)
# 0.50 = exact center
# 0.485 = slightly left
# 0.53 = slightly right
SPLIT_RATIO = 0.53

# Traffic thresholds (weighted score based)
LOW_THRESHOLD = 10      # score < 10  -> 10 sec green
MEDIUM_THRESHOLD = 20   # score < 20  -> 15 sec green
# score >= 20           -> 20 sec green

# Vehicle weights
VEHICLE_WEIGHTS = {
    'motorcycle': 0.5,
    'car': 1.0,
    'bus': 3.0,
    'truck': 3.0
}

# Timing constants
YELLOW_TIME = 2
ALL_RED_TIME = 1

LOG_FILE = "traffic_log.csv"

# ==========================================================
# CONNECT TO ARDUINO
# ==========================================================
print("Connecting to Arduino...")
arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
print("Arduino connected.")

# ==========================================================
# LOAD YOLO MODEL
# ==========================================================
print("Loading YOLO model...")
model = YOLO(MODEL_NAME)
print("Model loaded.")

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp",
            "Road A Count",
            "Road B Count",
            "Road A Score",
            "Road B Score",
            "Active Road",
            "Green Time",
            "Phase",
            "Emergency"
        ])

# ==========================================================
# VIDEO SOURCE
# ==========================================================
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Error: Could not open video source.")
    arduino.close()
    raise SystemExit

vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']

# ==========================================================
# STATE VARIABLES
# ==========================================================
current_road = 'A'          # Alternates A -> B -> A -> B
accident = False
emergency_road = 'A'

last_command = ""
next_allowed_time = 0

# For countdown display
active_road = None
cycle_start_time = 0

cv2.namedWindow("Smart Traffic System", cv2.WINDOW_NORMAL)

print("""
====================================================
SMART TRAFFIC SYSTEM STARTED

a -> Emergency on Road A
b -> Emergency on Road B
r -> Reset Emergency
ESC -> Exit
====================================================
""")


# ==========================================================
# FUNCTIONS
# ==========================================================
def bayesian_congestion(score):
    # Example priors
    prior_low = 0.33
    prior_medium = 0.33
    prior_high = 0.34

    # Simple likelihood model
    if score < 15:
        likelihood_low, likelihood_medium, likelihood_high = 0.8, 0.15, 0.05
    elif score < 30:
        likelihood_low, likelihood_medium, likelihood_high = 0.15, 0.7, 0.15
    else:
        likelihood_low, likelihood_medium, likelihood_high = 0.05, 0.2, 0.75

    post_low = prior_low * likelihood_low
    post_medium = prior_medium * likelihood_medium
    post_high = prior_high * likelihood_high

    total = post_low + post_medium + post_high

    post_low /= total
    post_medium /= total
    post_high /= total

    if post_low >= post_medium and post_low >= post_high:
        return "Low", 10
    elif post_medium >= post_high:
        return "Moderate", 15
    else:
        return "High", 20


def send_command(command, green_time=0):
    global last_command, next_allowed_time
    global current_road
    global active_road, cycle_start_time

    current_time = time.time()

    # Prevent sending new commands until current cycle finishes
    if current_time < next_allowed_time:
        return False

    arduino.write((command + '\n').encode())
    print("Sent:", command)

    last_command = command

    # Normal cycle command
    if command.startswith('A,') or command.startswith('B,'):
        cycle_duration = green_time + YELLOW_TIME + ALL_RED_TIME
        next_allowed_time = current_time + cycle_duration

        active_road = command[0]
        cycle_start_time = current_time

        # Alternate road for next cycle
        current_road = 'B' if current_road == 'A' else 'A'

    else:
        # Emergency / RESET
        next_allowed_time = current_time

    return True


def get_countdown():
    if active_road is None:
        return "IDLE", 0

    remaining = int(max(0, next_allowed_time - time.time()))

    if remaining > (YELLOW_TIME + ALL_RED_TIME):
        phase = "GREEN"
        seconds = remaining - (YELLOW_TIME + ALL_RED_TIME)
    elif remaining > ALL_RED_TIME:
        phase = "YELLOW"
        seconds = remaining - ALL_RED_TIME
    elif remaining > 0:
        phase = "ALL RED"
        seconds = remaining
    else:
        phase = "WAITING"
        seconds = 0

    return phase, seconds

def log_data(
    road_a_count,
    road_b_count,
    road_a_score,
    road_b_score,
    active_road,
    green_time,
    phase,
    emergency
):
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            road_a_count,
            road_b_count,
            round(road_a_score, 2),
            round(road_b_score, 2),
            active_road,
            green_time,
            phase,
            "ON" if emergency else "OFF"
        ])


# ==========================================================
# MAIN LOOP
# ==========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (1280, 720))
    h, w = frame.shape[:2]

    # Custom divider aligned to road median
    split_x = int(w * SPLIT_RATIO)

    # Detect vehicles
    results = model.track(
        frame,
        persist=True,
        imgsz=1280,
        verbose=False
    )

    road_a_score = 0.0
    road_b_score = 0.0
    road_a_count = 0
    road_b_count = 0

    # ------------------------------------------------------
    # PROCESS DETECTIONS
    # ------------------------------------------------------
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])

            if class_name not in vehicle_classes:
                continue

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2

            weight = VEHICLE_WEIGHTS.get(class_name, 1.0)

            if cx < split_x:
                road_a_count += 1
                road_a_score += weight
                color = (0, 255, 0)
            else:
                road_b_count += 1
                road_b_score += weight
                color = (255, 0, 0)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw center point
            cv2.circle(
                frame,
                ((x1 + x2) // 2, (y1 + y2) // 2),
                4,
                (0, 255, 255),
                -1
            )

            # Draw label
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    # ------------------------------------------------------
    # DECISION LOGIC
    # ------------------------------------------------------
    if accident:
        status = f"EMERGENCY - ROAD {emergency_road}"
        command = f"EMERGENCY,{emergency_road}"
        command_sent = send_command(command)
        green_time = 0

    else:
        scheduled_road = current_road

        if scheduled_road == 'A':
            selected_score = road_a_score
        else:
            selected_score = road_b_score

        congestion_level,green_time = bayesian_congestion(selected_score)

        status = (
            f"ROAD {scheduled_road} GREEN "
            f"({green_time}s) - {congestion_level}"
        )
        command = f"{scheduled_road},{green_time}"

        command_sent = send_command(command, green_time)


    # ------------------------------------------------------
    # COUNTDOWN
    # ------------------------------------------------------
    phase, seconds_left = get_countdown()
    
    # Log only when a normal GREEN command is sent
    if (
        command_sent
        and not accident
        and green_time > 0
        and phase == "GREEN"
    ):
        log_data(
            road_a_count=road_a_count,
            road_b_count=road_b_count,
            road_a_score=road_a_score,
            road_b_score=road_b_score,
            active_road=active_road,
            green_time=green_time,
            phase="GREEN",
            emergency=False
        )
    elif (
        command_sent
        and accident
    ):
        log_data(
            road_a_count=road_a_count,
            road_b_count=road_b_count,
            road_a_score=road_a_score,
            road_b_score=road_b_score,
            active_road=emergency_road,
            green_time=0,
            phase="EMERGENCY",
            emergency=True
        )
    # ------------------------------------------------------
    # DRAW SPLIT LINE
    # ------------------------------------------------------
    cv2.line(
        frame,
        (split_x, 0),
        (split_x, h),
        (0, 255, 255),
        3
    )

    # ------------------------------------------------------
    # ROAD LABELS
    # ------------------------------------------------------
    cv2.putText(
        frame,
        "ROAD A",
        (50, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        "ROAD B",
        (split_x + 60, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 0, 0),
        3
    )

    # ------------------------------------------------------
    # INFO PANEL (SMALLER)
    # ------------------------------------------------------
    panel_x1 = 10
    panel_y1 = 90
    panel_x2 = 500
    panel_y2 = 390

    cv2.rectangle(
        frame,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (0, 0, 0),
        -1
    )

    info_lines = [
        f"Road A Count : {road_a_count}",
        f"Road A Score : {road_a_score:.1f}",
        f"Road B Count : {road_b_count}",
        f"Road B Score : {road_b_score:.1f}",
        f"Status       : {status}",
        f"Active Road  : {active_road}",
        f"Phase        : {phase}",
        f"Countdown    : {seconds_left} sec",
        f"Emergency    : {'ON' if accident else 'OFF'}"
    ]

    y = 125
    for line in info_lines:
        cv2.putText(
            frame,
            line,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )
        y += 32

    # Emergency banner
    if accident:
        cv2.putText(
            frame,
            "EMERGENCY MODE",
            (700, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            4
        )

    # Show frame
    cv2.imshow("Smart Traffic System", frame)

    # ------------------------------------------------------
    # KEYBOARD CONTROLS
    # ------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    elif key == ord('a'):
        accident = True
        emergency_road = 'A'
        send_command("EMERGENCY,A")

    elif key == ord('b'):
        accident = True
        emergency_road = 'B'
        send_command("EMERGENCY,B")

    elif key == ord('r'):
        accident = False
        send_command("RESET")
        last_command = ""

# ==========================================================
# CLEANUP
# ==========================================================
cap.release()
cv2.destroyAllWindows()
arduino.close()
print("System Closed Successfully.")