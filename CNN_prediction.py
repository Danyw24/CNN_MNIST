import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
console = Console()
# Cargar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Cargar el modelo guardado
model = tf.keras.models.load_model('CNN_MNIST.keras')

# Elegir una imagen de los datos de prueba
index = 9  # Se puede cambiar el índice para probar diferentes imágenes
image = x_test[index]
label = y_test[index]

# Mostrar la imagen original
plt.imshow(image, cmap='gray')
plt.title(f'Imagen de prueba - Etiqueta real: {label}')
plt.axis('off')
plt.show()

# Preprocesar la imagen de la misma manera que se entrenó el modelo
image = image / 255.0  # Normalizar los valores de píxeles (0-255 -> 0-1)
image = image.reshape(1, 28, 28, 1)  # Reshape para (1, 28, 28, 1)

# Hacer una predicción
prediction = model.predict(image)
predicted_label = np.argmax(prediction)

# Mostrar la predicción

console.print(f'[bold cyan]Predicción: {predicted_label}[/bold cyan]')

