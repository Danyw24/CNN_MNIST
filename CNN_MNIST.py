# Author: Danyw24 - 09/03/2025
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from rich.console import Console
from rich.table import Table
from rich.progress import track
import numpy as np

console = Console()

console.print("[bold green][+] Ejemplo de red neuronal simple[/bold green]")

# 1. Cargar y preparar los datos MNIST (60000 imágenes de entrenamiento y prueba)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

console.print(f"[bold green][+]Número de imágenes de entrenamiento: {len(x_train)}[/bold green]")

# Ver la estructura de los datos de entrenamiento
table = Table(show_header=True, header_style="bold magenta")
table.add_column("X Train Shape", justify="right")
table.add_column("Y Train Shape", justify="right")
table.add_row(str(x_train.shape), str(y_train.shape))
console.print(table)

# Ejemplo de una imagen y su etiqueta
example_image = x_train[0]  # Primera imagen de entrenamiento
example_label = y_train[0]  # Etiqueta de la primera imagen

console.print(f"[bold cyan]Mostrando imagen de ejemplo con etiqueta: {example_label}[/bold cyan]")

# Mostrar la imagen y su etiqueta
plt.imshow(example_image, cmap='gray')
plt.title(f"Etiqueta: {example_label}")
plt.axis('off')  # Ocultar ejes
plt.show()

# Función para mostrar imágenes de entrenamiento
def plot_images(images, labels, num_images=25) -> None:
    """
    Función para mostrar imágenes de entrenamiento
    
    Parameters:
    images (numpy.ndarray): Matriz de imágenes de entrenamiento
    labels (numpy.ndarray): Matriz de etiquetas de las imágenes     
    num_images (int): Número de imágenes a mostrar
    """
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i].squeeze(), cmap=plt.cm.binary)
        plt.xlabel(labels[i])
    plt.show()

# Llamar a la función para mostrar las primeras 25 imágenes de entrenamiento
plot_images(x_train, y_train)


# Normalizar los valores de píxeles (0-255 -> 0-1)
"""
Normalizar los valores de píxeles (0-255 -> 0-1) sirve para 
que los valores de los píxeles sean más fáciles de tratar y
procesar en el modelo.  
"""
x_train = x_train / 255.0
x_test = x_test / 255.0


# Reformatear los datos para la red neuronal (agregar dimensión de canal)
"""
Reformatear los datos para que la red neuronal pueda procesar los datos de entrada.
Los datos de entrada se representan como una matriz de imágenes de 28x28 píxeles en escala de grises.
y se agregan una dimensión de canal para representar cada imagen como una matriz de píxeles.
"""
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 2. Construir el modelo
"""
Explicación de la arquitectura de la red neuronal:

- La capa Flatten es una capa plana que convierte los datos de entrada 
en un vector unidimensional. Esto es útil para que la capa oculta pueda 
procesar los datos de entrada de manera eficiente, es decir (28x28x1) -> (784).

- la capa Dense oculta con 128 neuronas y activación ReLU. Esta capa oculta 
procesa los datos de entrada y produce una salida de tamaño 128 ya que es una capa
conectada con todas las neuronas de la capa de entrada, esta maneja 784 * 128 más 128 sesgos
en total 100480 parámetros.

- La capa Dropout es una capa de regularización que elimina una proporción de las neuronas
de la capa oculta, esto ayuda a regularizar el modelo y a prevenir la memoria excesiva.
en este caso, elimina el 20% de las neuronas de la capa oculta de manera aleatoria.

- La capa de salida con 10 neuronas (una por cada dígito) y activación softmax. Esta capa
produce una salida de tamaño 10 ya que es una capa conectada con todas las neuronas de la capa
oculta, esto maneja 100480 parámetros con 10 salidas ( 0 al 9 ) de clasificación. 

"""
model = tf.keras.Sequential([
    # Capa de entrada (imágenes 28x28 píxeles en escala de grises)
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    
    # Capa oculta con 128 neuronas y activación ReLU
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Capa de Dropout para regularización
    tf.keras.layers.Dropout(0.2), # %20 de la capa oculta se elimina aleatoriamente
    
    # Capa de salida con 10 neuronas (una por cada dígito) y activación softmax
    tf.keras.layers.Dense(10, activation='softmax')
])


from tensorflow.keras.utils import plot_model

# Asegúrate de que el modelo esté definido y compilado
plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)
# 3. Compilar el modelo
"""
- La optimización se utiliza para ajustar los pesos del bias de parámetros de la red neuronal, mediante
el método de descenso de gradiente y el optimizador Adam, que es una 
variante de RMSProp + SGD(descenso de gradiente estocástico), RMSProp es una optimización que ajusta los pesos
de manera dinamica con el fin de mejorar el rendimiento del modelo en el entrenamiento.

- loss: utiliza sparse_categorical_crossentropy para calcular la diferencia entre las salidas y las etiquetas
ejemplo: si la red predice [0.1, 0.8, 0.1]] y la etiqueta es [0, 1, 0] entonces el loss será calculado como:
loss = tf.nn.sparse_categorical_crossentropy(y_true=[0, 1, 0], y_pred=[0.1, 0.8, 0.1]) o  L=−log(0.8)
entonces L=0.223  

- metrics: utiliza accuracy para calcular la precisión del modelo en el conjunto de prueba y transformar el resultado
en un porcentaje 0.8 -> 80%
"""
model.compile(optimizer='adam', #RMSProp + SGD 
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])



# 4. Entrenar el modelo
console.print("[bold green][+] Entrenamiento iniciado...[/bold green]")

history = model.fit(x_train, y_train,
    epochs=15,  # epocas de entrenamiento
    validation_split=0.2,  # %20 de los datos de entrenamiento se utilizan para validación
)

# 5. Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=3)
print(f'\nPrecisión en datos de prueba: {test_acc:.4f}')

# Mostrar resumen al final del entrenamiento
console.print("\n[bold blue]Resultados del entrenamiento:[/bold blue]")

# Mostrar precisión final y pérdida final en la consola
final_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
final_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

console.print(f"[bold yellow]Precisión de entrenamiento final: {final_accuracy:.4f}[/bold yellow]")
console.print(f"[bold yellow]Precisión de validación final: {final_val_accuracy:.4f}[/bold yellow]")
console.print(f"[bold yellow]Pérdida de entrenamiento final: {final_loss:.4f}[/bold yellow]")
console.print(f"[bold yellow]Pérdida de validación final: {final_val_loss:.4f}[/bold yellow]")

# Guardar el modelo completo
model.save('CNN_MNIST.keras')

# Guardar solo los pesos
model.save_weights('CNN_MNIST_.weights.h5')
console.print(f"[bold yellow][+] Modelo y pesos Guardados/bold yellow]")


# 6. Hacer una predicción de ejemplo
sample_index = 0  # Cambiar este índice para probar diferentes ejemplos
sample_image = x_test[sample_index]
prediction = model.predict(np.array([sample_image]))
predicted_label = np.argmax(prediction)

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# Mostrar resultados
plt.figure()
plt.imshow(sample_image.squeeze(), cmap=plt.cm.binary)
plt.title(f'Predicción: {predicted_label}, Real: {y_test[sample_index]}')
plt.colorbar()
plt.grid(False)
plt.show()

# Gráfico de precisión durante el entrenamiento
plt.plot(history.history['accuracy'], label='Precisión entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()