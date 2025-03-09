# Reconocimiento de Dígitos Manuscritos mediante Redes Neuronales Profundas 
``Autor: Danyw24``

Este repositorio implementa un sistema de clasificación de dígitos manuscritos basado en redes neuronales, utilizando el conjunto de datos MNIST como referencia estándar en la literatura de aprendizaje profundo. La arquitectura del modelo se fundamenta en un perceptrón multicapa (MLP) con una estructura optimizada para el reconocimiento de patrones visuales, siguiendo los principios formales del aprendizaje basado en gradientes descritos en el artículo **"Gradient-Based Learning Applied to Document Recognition"** de Yann LeCun et al.

## 📌 **Requisitos del Entorno de Ejecución**

Para garantizar la correcta ejecución del modelo, se requiere el siguiente entorno de desarrollo:

- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib (opcional para visualización de resultados)

Instalación de dependencias mediante pip:

```bash
pip install tensorflow numpy matplotlib
```

## 📂 **Estructura Modular del Proyecto**

# 📁 Estructura del Proyecto MNIST-CNN

```bash
├── dvenv/            # Entorno virtual Python
├── CNN_MNIST.py      # Arquitectura principal de la CNN
├── CNN_prediction.py # Script para hacer predicciones
│
├── models/           # Directorio de modelos guardados
│   ├── CNN_MNIST.keras         # Modelo completo guardado
│   └── CNN_MNIST_.weights.h5   # Pesos del modelo
│
├── docs/             # Documentación y visualizaciones
│   ├── model.png            # Diagrama arquitectura del modelo
│   ├── model_structure.png  # Estructura detallada en texto
│   └── Training_accuracy.png # Gráfico de precisión

```

## 📊 **Preprocesamiento y Normalización de Datos**

![image](https://github.com/user-attachments/assets/2641c825-2fb9-47a5-96ed-14b328a34ea5)




### **Normalización de Características**

Cada imagen del conjunto MNIST presenta valores de intensidad en el rango `[0,255]`. Para mejorar la estabilidad numérica y la eficiencia del entrenamiento, se normalizan los valores en el intervalo `[0,1]` mediante:

```python
"""
Normalizar los valores de píxeles (0-255 -> 0-1) sirve para 
que los valores de los píxeles sean más fáciles de tratar y
procesar en el modelo.  
"""
x_train = x_train / 255.0
x_test = x_test / 255.0

```


### **Transformación Espacial**

Reformatear los datos para que la red neuronal pueda procesar los datos de entrada.
Los datos de entrada se representan como una matriz de imágenes de 28x28 píxeles en escala de grises.
y se agregan una dimensión de canal para representar cada imagen como una matriz de píxeles.

```python
X = X.reshape(-1, 28, 28, 1)
```

Esta operación preserva la información espacial y permite la explotación de correlaciones estructurales en la entrada.

---

## 🏗️ **Arquitectura del Modelo Neuronal**

![image](https://github.com/user-attachments/assets/498a4f6d-b61b-46af-ba14-791ceb216cd2)



La arquitectura implementada es un perceptrón multicapa con una única capa oculta densa:

| Capa            | Dimensión de Entrada | Número de Parámetros |
| --------------- | ------------------- | ------------------- |
| Flatten         | (28,28) → 784       | 0                 |
| Dense (ReLU)    | 784 → 128           | 100480            |
| Dense (Softmax) | 128 → 10            | 1290              |

### **Funciones de Activación**

- **ReLU (Rectified Linear Unit)**: Proporciona una activación no saturante para mitigar el problema del desvanecimiento del gradiente.
- 
![image](https://github.com/user-attachments/assets/11aef82e-4310-4b88-8387-147ba95e0819)


- **Softmax**: Convierte la salida de la capa final en una distribución de probabilidad sobre las 10 clases posibles con valores entre 0 y 1.

![image](https://github.com/user-attachments/assets/7ef664e5-71c4-48e3-b37d-b0f3a5dceb1a)

---

## ⚙️ **Función de Pérdida y Algoritmo de Optimización**

La función de pérdida empleada es **Sparse Categorical Crossentropy**, la cual mide la divergencia entre la distribución de las predicciones y las etiquetas reales:

![image](https://github.com/user-attachments/assets/ffafe3ab-05c0-4328-b098-3eb0cd021921)

 loss: utiliza sparse_categorical_crossentropy para calcular la diferencia entre las salidas y las etiquetas
ejemplo: si la red predice [0.1, 0.8, 0.1]] y la etiqueta es [0, 1, 0] entonces el loss será calculado como:
loss = tf.nn.sparse_categorical_crossentropy(y_true=[0, 1, 0], y_pred=[0.1, 0.8, 0.1]) o  L=−log(0.8)
entonces **L=0.223**  


El algoritmo de optimización utilizado es **Adam**, que integra los métodos de descenso de gradiente estocástico (SGD) y RMSProp para una convergencia más eficiente,
La optimización se utiliza para ajustar los pesos del bias de parámetros de la red neuronal, mediante
el método de descenso de gradiente y el optimizador Adam, que es una 
variante de RMSProp + SGD(descenso de gradiente estocástico), RMSProp es una optimización que ajusta los pesos
de manera dinamica con el fin de mejorar el rendimiento del modelo en el entrenamiento.

---

## 📈 **Métricas de Evaluación del Modelo**

El desempeño del modelo se cuantifica utilizando la métrica de **precisión (accuracy)**, definida como:


Para evaluar el modelo entrenado:

```python
loss, acc = model.evaluate(X_test, y_test)
print(f"Precisión obtenida: {acc*100:.2f}%")
```

---

## 🚀 **Ejemplo de Implementación del Entrenamiento**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Definición de la arquitectura del modelo
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Configuración del modelo
model.compile(
    optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Entrenamiento del modelo
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

---

## 🎯 **Resultados Experimentales**

![image](https://github.com/user-attachments/assets/64f4e085-c6d0-4ddf-9598-52d2f82523a2)


![image](https://github.com/user-attachments/assets/1896780b-34ec-4d8c-a8c7-63b5cdd217c3)


## 📊 **Métricas de Rendimiento**

| Métrica               | Entrenamiento | Validación |
|-----------------------|---------------|------------|
| **Precisión**         | 98.86%        | 97.68%     |
| **Pérdida**           | 0.0339        | 0.0943     |

## 🚀 **Resultado Final en Pruebas**
```python
Precisión en datos de prueba: 0.9780
```
---

## 📜 **Referencias Bibliográficas**

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition*. *Proceedings of the IEEE*.
- Documentación oficial de Keras: [https://keras.io](https://keras.io)
- Recursos de TensorFlow: [https://www.tensorflow.org](https://www.tensorflow.org)

---


