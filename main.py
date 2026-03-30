import cv2

cap = cv2.VideoCapture("traffic_video1.mp4")

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)

    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    vehicle_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 300:   # IMPORTANT change (not 500)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            vehicle_count += 1

    if vehicle_count < 5:
        signal = "GREEN: 10 sec"
    elif vehicle_count < 15:
        signal = "GREEN: 20 sec"
    else:
        signal = "GREEN: 30 sec"

    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    cv2.putText(frame, signal, (20,100),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    
    cv2.imshow("Detection", frame)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()