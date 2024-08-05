import cv2
import os

# Lista de nombres de los videos (sin la extensión .mp4)
videos = [
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

# Crear carpeta principal 'class/data' si no existe
os.makedirs('class/data', exist_ok=True)

for video in videos:
    # Crear una subcarpeta para cada clase si no existe
    os.makedirs(f'class/data/{video}', exist_ok=True)

    # Leer el video desde la carpeta 'class'
    vidcap = cv2.VideoCapture(f'class/{video}.mp4')
    success, image = vidcap.read()
    count = 0

    while success:
        # Guardar cada frame como una imagen JPEG en la subcarpeta correspondiente
        cv2.imwrite(f'class/data/{video}/frame{count}.jpg', image)
        success, image = vidcap.read()
        print(f'Read a new frame from {video}: ', success)
        count += 1

    # Liberar recursos
    vidcap.release()
