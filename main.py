import os
import cv2
import keras
import numpy as np
import progressbar
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

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

num_classes = len(classes)
img_rows, img_cols = 64, 64  # Puedes ajustar el tamaño según la resolución de tus imágenes

def load_data():
    data = []
    target = []
    for index, class_ in enumerate(classes):
        folder_path = os.path.join('class/data', class_)  # Actualiza la ruta
        print(f"Normalizing {folder_path}")
        images = os.listdir(folder_path)
        with progressbar.ProgressBar(max_value=len(images), prefix=f"Processing {class_}: ") as bar:
            for i, img in enumerate(images):
                img_path = os.path.join(folder_path, img)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Leer en color
                    img = cv2.resize(img, (img_rows, img_cols))
                    data.append(np.array(img))
                    target.append(index)
                except Exception as e:
                    print(f"Error reading file {img_path}: {e}")
                    continue
                bar.update(i + 1)
    data = np.array(data)
    target = np.array(target)
    new_target = keras.utils.to_categorical(target, num_classes)
    return data, new_target


data, target = load_data()

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.1)

model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
history = model.fit(
    train_data,
    train_target,
    batch_size=32,
    epochs=3,
    verbose=1,
    validation_data=(test_data, test_target)
)

model.save('model.h5')

# Evaluar el modelo
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(test_target, axis=1)
confusion_mtx = sklearn.metrics.confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 10))
sns.heatmap(confusion_mtx, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.title('Confusion Matrix')
plt.savefig('graficas/confusion_matrix.png')
plt.close()

# Crear y guardar la gráfica de precisión
plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('graficas/accuracy_plot.png')
plt.close()

print(history.history['val_loss'])

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.title('Gráfica de Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.savefig('graficas/loss_plot.png')
plt.close()