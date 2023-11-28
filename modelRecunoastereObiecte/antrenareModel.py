import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(train_images, train_labels), _ = keras.datasets.cifar10.load_data()

train_images = train_images / 255.0

model = keras.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

# compileaza modelul
model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# antreneaza modelul
model.fit(train_images, train_labels, epochs=10)