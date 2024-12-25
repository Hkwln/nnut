import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

# Load the dataset
mnist_dataset = tfds.load(name = "mnist", split = ["train""test"], shuffle_files = True, as_supervised=True, with_info=True)
# Split the dataset into training and testing and the belonging labels
(x_train, y_train), (x_test, y_test) = mnist_dataset
# Normalize the data
x_train = x_train /255.0
x_test = x_test /255.0 

# build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Droupout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
predictions = model(x_train[:1]).numpy()
predictions
tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)