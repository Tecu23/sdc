from flask import Flask, render_template, Response, request, send_from_directory
import cv2

from camera import VideoCamera
from utilities import fromFrameToJPEG

from lane_following.lane_following import detect_lane
# from sign_detection.sign_detection import detect_Signs

# from drive import Drive_Car

pi_camera = VideoCamera(flip=False) # flip pi camera if upside down.

# App Globals (do not edit)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') #you can customze index.html here

def gen(camera):
    #get camera frame
    while True:
        frame = camera.get_frame()

        distance, curvature, out_frame = detect_lane(frame)

        # 2. Detecting and extracting information from the signs
        # Mode , Tracked_class, out = detect_Signs(frame, out_frame)

        # Current_State = [distance, curvature , out , Mode , Tracked_class]
        # output = Drive_Car(Current_State)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + fromFrameToJPEG(out_frame) + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(pi_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=False)