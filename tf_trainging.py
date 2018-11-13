import tensorflow as tf
from matplotlib import pyplot as plt
import time

training_csv = 'training_set.csv'
training_dataset = tf.data.experimental.make_csv_dataset(training_csv, batch_size=32)
iter = training_dataset.make_one_shot_iterator()
next = iter.get_next()
print(next)  # next is a dict with key=columns names and value=column data
'''inputs, labels = next['text'], next['sentiment']
with tf.Session() as sess:
    sess.run([inputs, labels])'''
