import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

# Load the dataset
mnist_dataset = tfds.load(name = "mnist", split = ["train""test"], shuffle_files = True, as_supervised=True, with_info=True)


# Define the model
