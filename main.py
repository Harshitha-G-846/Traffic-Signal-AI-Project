import cv2

# Load video
cap = cv2.VideoCapture("traffic_video3.mp4")

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

# Accident flag
accident = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Threshold to remove shadows/noise
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    vehicle_count = 0

    # Draw bounding boxes
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 800:  # Adjust this if needed
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            vehicle_count += 1

    # Signal logic
    if accident:
        signal = "EMERGENCY GREEN"
    elif vehicle_count < 5:
        signal = "GREEN: 10 sec"
    elif vehicle_count < 15:
        signal = "GREEN: 20 sec"
    else:
        signal = "GREEN: 30 sec"

    # Display info
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, signal, (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame, "Smart Traffic System", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Accident alert display
    if accident:
        cv2.putText(frame, "ACCIDENT DETECTED!", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show frame
    cv2.imshow("Traffic Detection System", frame)

    # SINGLE waitKey (IMPORTANT)
    key = cv2.waitKey(30) & 0xFF

    if key == 27:  # ESC key to exit
        break

    if key == ord('a'):  # Press 'A' to simulate accident
        accident = True

    if key == ord('r'):  # Press 'R' to reset accident
        accident = False

# Release resources
cap.release()
cv2.destroyAllWindows()