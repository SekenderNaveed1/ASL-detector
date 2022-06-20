from flask import Flask, render_template
from turbo_flask import Turbo
import threading
import time
import cv2
import detect
from datetime import datetime
from flask import Response

app = Flask(__name__)
turbo = Turbo(app)

camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('homepage.html')

@app.context_processor
def inject_load():
    varName = detect.return_value
    return {'load1': varName}

@app.before_first_request
def before_first_request():
    threading.Thread(target=update_load).start()

def update_load():
    with app.app_context():
        while True:
            time.sleep(2)
            turbo.push(turbo.replace(render_template('outputbox.html'), 'load'))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

if __name__ == '__main__':
    app.run(debug = True)
