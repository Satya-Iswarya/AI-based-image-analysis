import cv2
# Open default camera (index 0 is primary webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access camera")
    exit()

print("✅ Camera opened successfully. Press 'q' to quit.")

while True: # infinite loop runs until we press q
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    cv2.imshow("Camera Test", frame) # shows frame in a window

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
