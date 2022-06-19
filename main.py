from flask import Flask, render_template, Response, request, send_from_directory
from camera import VideoCamera
from utilities import fromFrameToJPEG
import os
import cv2
import numpy as np

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
        # HERE WILL BE THE CONTROL

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150, None)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + fromFrameToJPEG(canny) + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(pi_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=False)