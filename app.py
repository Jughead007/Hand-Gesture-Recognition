import cv2
import numpy as np
from flask import Flask, Response, render_template
from main_controller import MainController

app = Flask(__name__)

# Initialize controller once
controller = MainController(
    detection_model='models/hand_detector.onnx',
    classification_model='models/crops_classifier.onnx'
)

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Process frame
        bboxes, ids, labels = controller(frame)
        
        # Draw results
        if bboxes is not None:
            bboxes = bboxes.astype(np.int32)
            for i in range(bboxes.shape[0]):
                box = bboxes[i, :]
                gesture = labels[i] if labels[i] is not None else "None"
                
                # Draw bounding box and label
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
                cv2.putText(
                    frame,
                    f"ID {ids[i]} : {gesture}",
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)