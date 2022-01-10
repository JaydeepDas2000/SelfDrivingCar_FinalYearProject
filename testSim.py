print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()

app = Flask(__name__) # '__main__'
maxSpeed = 15
#minSpeed = 10
#speed_limit = maxSpeed

def preProcessing(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(200,66))
    img = img/255
    
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preProcessing(image)
    image = np.asarray([image])
    steering = float(model.predict(image))
    throttle = 1.0 - speed / maxSpeed
    #global speed_limit
    #if speed > speed_limit:
    #    speed_limit = minSpeed
    #else:
    #    speed_limit = maxSpeed
    #throttle = 1.0 - steering**2 - (speed/speed_limit)**2
    
    print('{} {} {}'.format(steering, throttle, speed))
    sendControl(steering, throttle)

@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)

def sendControl(steering, throttle):
    sio.emit('steer', data = {
        'steering_angle' : steering.__str__(),
        'throttle' : throttle.__str__()
    })

if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)