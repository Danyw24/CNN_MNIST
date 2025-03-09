import tensorflow as tf 
import numpy as np

class NTMMemory(tf.keras.layers.Layer):
    # Constructor permanece igual

    def call(self, control_output, MEMORY):
        # Dividir el control_output adecuadamente
        # Ajusta las dimensiones basado en lo que requiere tu diseño
        # Asegúrate de que tiene sentido con [memory_size, memory_vector_dim]
        num_units_used = self.memory_vector_dim * 2
        write_weights = control_output[:, :self.num_outputs]  # Ajusta esto si es necesario
        erase_vector = control_output[:, self.num_outputs:self.num_outputs + num_units_used]
        add_vector = control_output[:, self.num_outputs + num_units_used:]

        # Ahora puedes verificar y ajustar las formas para que estas operaciones tengan sentido
        return write_weights, erase_vector, add_vector

def write_memory(memory, write_weights, erase_vector, add_vector):
    # Asegúrate de que erase_vector y add_vector tengan formas que interactúen bien con la memoria
    erase_vector = tf.reshape(erase_vector, [-1, memory.shape[1]])
    add_vector = tf.reshape(add_vector, [-1, memory.shape[1]])

    erase = tf.reduce_sum(write_weights[:, :, tf.newaxis] * erase_vector[:, tf.newaxis, :], axis=1)
    add = tf.reduce_sum(write_weights[:, :, tf.newaxis] * add_vector[:, tf.newaxis, :], axis=1)
    return memory * (1 - erase) + add
        
class NTM(tf.keras.Model):
    def __init__(self, num_inputs, num_outputs, memory_size, memory_vector_dim):   
        super(NTM, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.memory_layer = NTMMemory(memory_size)  # Mueve la inicialización aquí

        # Inicializar memoria
        self.memory = tf.Variable(
            tf.zeros([memory_size, memory_vector_dim]),
        )

    def call(self, inputs):
        # Pasar la entrada a la red de control
        control_output = NTMController(self.num_inputs, self.num_outputs, self.memory_vector_dim)(inputs, self.memory)
        
        # Usar la instancia ya inicializada de NTMMemory
        memory_output = self.memory_layer(control_output)
        
        # Actualizar memoria
        self.memory = write_memory(self.memory, memory_output[0], memory_output[1], memory_output[2])
        
        return memory_output


class NTMController(tf.keras.layers.Layer):
    """La arquitectura de la red de control de la NTM
    
    La red de control recibe como entradas las salidas de la red de memoria y las 
    salidas de la red de lectura. La salida de la red de control es una tupla de 
    tres tensores: la salida de la red de memoria, la salida de la red de lectura 
    y la salida de la red de control.
    """
    def __init__(self, num_inputs, num_outputs, memory_vector_dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_outputs + memory_vector_dim * 2 )

    def call(self, inputs, prevs_state):
        batch_size = tf.shape(inputs)[0]
    
        # Asegúrate de que prevs_state tenga un tamaño batch igual al de inputs
        prevs_state = tf.expand_dims(prevs_state, axis=0)
        prevs_state = tf.tile(prevs_state, [batch_size, 1, 1])
    
        x = tf.concat([inputs, prevs_state], axis=-1)
        x = self.dense1(x)
        return self.dense2(x)


# Mecanismos de Addressing 

def cosine_similarity(x, y):
    """
     1e-8 (0.00000001) -> al utilizar esta notación, se evita que la formula de cero
     en todas sus entradas, al agregar 1e-8 se asegura que el denominador no sea cero
    """
    return tf.reduce_sum( x*y, axis=-1) / ( tf.norm(x, axis=-1) * tf.norm(y, axis=-1) + 1e-8)

def content_addressing(key, memory):
    """
    key pasa de tener la forma [batch_size, features] a [batch_size, 1, features].
    y posteriormente se aplica el producto de matriz de coseno similaridad,
    despues retorna el producto de la matriz de coseno de similaridad y la
    pasa por una capa softmax para que la salida sea un vector de probabilidades.
    entre 0 y 1.
    """
    similatiry = cosine_similarity(key[:, tf.newaxis, :], memory) 
    return tf.nn.softmax(similatiry, axis=-1) 

def read_memory(memory, read_weights):
    """
    Lee la memoria usando pesos de lectura.

    Realiza una multiplicación elemento a elemento entre la memoria
    y los pesos de lectura (ajustados en dimensiones) y luego suma
    a través de la dimensión de memoria para obtener un vector de
    salida ponderado según los pesos de lectura.
    """
    return tf.reduce_sum(memory * read_weights[:, :, tf.newaxis], axis=1)

def write_memory(memory, write_weights, erase_vector, add_vector):
    erase = tf.reduce_sum(write_weights[:, :, tf.newaxis] * erase_vector[:, tf.newaxis, :], axis=1)
    add = tf.reduce_sum(write_weights[:, :, tf.newaxis] * add_vector[:, tf.newaxis, :], axis=1)
    return memory * (1 - erase) + add

@tf.function
def train_step(inputs, targets, model, optimizer):
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = tf.reduce_mean(tf.square(outputs - targets))
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def generate__task(sequence_length, vector_dim):
    sequence = np.random.randint(0, 2, size=(1, sequence_length, vector_dim))
    inputs = np.concatenate([sequence, np.zeros((1, 1, vector_dim))], axis=1)
    targets = np.concatenate([np.zeros_like(sequence), sequence], axis=1)
    return inputs, targets



# Inicializa el optimizador
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Uso de la función generate__task y el bucle de entrenamiento
inputs, targets = generate__task(10, 8)
ntm = NTM(num_inputs=8, num_outputs=8, memory_size=11,   memory_vector_dim=20)

# Ejecuta una iteración de entrenamiento
loss = train_step(inputs, targets, ntm, optimizer)
print(f'Loss: {loss.numpy()}')







