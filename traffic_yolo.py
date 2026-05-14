import cv2
from ultralytics import YOLO
import serial
import time

# ==================================
# SETTINGS
# ==================================
ARDUINO_PORT = 'COM5'      # Change to your COM port
BAUD_RATE = 9600
VIDEO_SOURCE = 'traffic_video1.mp4'  # Use 0 for webcam
MODEL_NAME = 'yolov8s.pt'
CONFIDENCE_THRESHOLD = 0.35

# Traffic thresholds
LOW_THRESHOLD = 20      # 0-19 vehicles -> 10 sec green
MEDIUM_THRESHOLD = 45   # 20-44 vehicles -> 15 sec green
# 45+ vehicles -> 20 sec green

# ==================================
# CONNECT TO ARDUINO
# ==================================
print('Connecting to Arduino...')
arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
print('Arduino connected.')

# ==================================
# LOAD MODEL
# ==================================
print('Loading YOLO model...')
model = YOLO(MODEL_NAME)
print('Model loaded.')

# ==================================
# VIDEO SOURCE
# ==================================
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print('Could not open video source.')
    arduino.close()
    raise SystemExit

vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']

accident = False
last_command = ''
last_send_time = 0
SEND_INTERVAL = 1.0

cv2.namedWindow('YOLO Traffic Detection System', cv2.WINDOW_NORMAL)


def send_command(command):
    global last_command, last_send_time

    now = time.time()
    if command != last_command or (now - last_send_time) > SEND_INTERVAL:
        arduino.write((command + '\n').encode())
        print(f'Sent: {command}')
        last_command = command
        last_send_time = now


print('=====================================')
print('YOLO SMART TRAFFIC SYSTEM STARTED')
print('A -> Emergency Mode')
print('R -> Reset Emergency Mode')
print('ESC -> Exit')
print('=====================================')


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    results = model.track(
        frame,
        persist=True,
        imgsz=1280,
        verbose=False
    )

    vehicle_count = 0
    car_count = 0
    bike_count = 0
    bus_count = 0
    truck_count = 0

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])

            if class_name in vehicle_classes and confidence >= CONFIDENCE_THRESHOLD:
                vehicle_count += 1

                if class_name == 'car':
                    car_count += 1
                elif class_name == 'motorcycle':
                    bike_count += 1
                elif class_name == 'bus':
                    bus_count += 1
                elif class_name == 'truck':
                    truck_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f'{class_name} {confidence:.2f}',
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2,
                )

    # ==================================
    # DECISION LOGIC
    # ==================================
    if accident:
        status = 'EMERGENCY MODE'
        command = 'EMERGENCY'

    elif vehicle_count < LOW_THRESHOLD:
        status = 'LOW TRAFFIC (10s)'
        command = 'LOW,10'

    elif vehicle_count < MEDIUM_THRESHOLD:
        status = 'MEDIUM TRAFFIC (15s)'
        command = 'MEDIUM,15'

    else:
        status = 'HIGH TRAFFIC (20s)'
        command = 'HIGH,20'

    send_command(command)

    # Display panel
    cv2.rectangle(frame, (10, 10), (520, 320), (0, 0, 0), -1)

    info = [
        f'Vehicles : {vehicle_count}',
        f'Cars     : {car_count}',
        f'Bikes    : {bike_count}',
        f'Buses    : {bus_count}',
        f'Trucks   : {truck_count}',
        f'Status   : {status}',
    ]

    y = 40
    for line in info:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        y += 40

    if accident:
        cv2.putText(
            frame,
            'ACCIDENT DETECTED!',
            (560, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )

    cv2.imshow('YOLO Traffic Detection System', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    elif key == ord('a'):
        accident = True
        send_command('EMERGENCY')

    elif key == ord('r'):
        accident = False
        send_command('RESET')

# Cleanup
cap.release()
cv2.destroyAllWindows()
arduino.close()
print('System Closed Successfully.')