from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import cv2
import keras
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
socketio = SocketIO(app)
model = keras.models.load_model('model.h5')
img_rows, img_cols = 64, 64
# Tamaño de las imágenes

# Lista de figuras geométricas
classes = [
    'circulo',
    'triangulo',
    'cuadrado',
    'rectangulo',
    'pentagono',
    'hexagono',
    'octagono',
    'ovalo',
    'estrella',
    'cruz',
    'flecha',
    'rombo',
    'trapecio'
]

def preprocess_image(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (img_rows, img_cols))
    img = img.reshape(1, img_rows, img_cols, 3)
    img = img.astype('float32') / 255.0
    return img

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('frame')
def handle_frame(data):
    # Decode base64 image
    header, encoded = data.split(",", 1)
    nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process image
    img = preprocess_image(img)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    class_name = classes[class_idx]

    # Send prediction to client
    emit('prediction', class_name)

#hola
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)

