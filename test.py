import cv2

print("[TEST] Trying to open camera at index 0")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows-specific fix

if not cap.isOpened():
    print("[TEST] Failed to open camera at index 0. Trying others...")
    for i in range(1, 5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"[TEST] Opened camera at index {i}")
            break
    else:
        print("[TEST] No camera could be opened.")
        exit()

print("[TEST] Camera opened. Starting feed. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[TEST] Failed to grab frame.")
        break

    cv2.imshow("Test Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[TEST] 'q' pressed. Exiting.")
        break

cap.release()
cv2.destroyAllWindows()
