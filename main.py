import cv2

cap=cv2.VideoCapture("traffic_video1.mp4")

while True: 
    ret, frame= cap.read()
    if not ret:
        break
    cv2.imshow("Traffic Video",frame)

    if cv2.waitKey(1) == 27:
        break
    
cap.release()
cv2.destroyAllWindows()