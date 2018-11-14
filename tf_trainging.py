import tensorflow as tf
from matplotlib import pyplot as plt
import time

training_csv = 'training_set.csv'
training_dataset = tf.data.experimental.make_csv_dataset(training_csv, batch_size=32)
iter = training_dataset.make_one_shot_iterator()
print(iter.get_next())
print(iter.get_next())

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 4  # amount of majors
