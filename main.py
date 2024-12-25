import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

# Load the dataset
mnist_dataset = tfds.load(name = "mnist", split = ['train','test'], shuffle_files = True, as_supervised=True, with_info=True)
# Split the dataset into training and testing and the belonging labels

def prepare_data(dataset):
  images, labels = [], []
  for image, label in dataset.as_numpy_iterator():
    image, label = np.array(image) / 255.0, np.eye(10)[label]  # Normalize the image and one-hot encode the label
    image = np.array(image)  # Ensure image is a numpy array
    image = image.reshape(-1)  # Reshape to (28, 28, 1)
    images.append(image)
    labels.append(label)
  return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)

# Prepare the training and testing data
x_train, y_train = prepare_data(mnist_dataset['train'])
x_test, y_test = prepare_data(mnist_dataset['test'])

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