import tensorflow as tf
import numpy as np
import csv
import datetime

users_prediction = {}
z = []
with open('validation_set.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile)
    itercsv = iter(spamreader)
    next(itercsv)
    for row in spamreader:
        users_prediction[row[0]] = None
        z.append([float(i) for i in row[2:]])
z_data = np.array(z)

w = []
with open('major_index.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile)
    itercsv = iter(spamreader)
    next(itercsv)
    for row in spamreader:
        user = row[0]
        if user in users_prediction:
            w.append([float(j) for j in row[2:]])
w_data = np.array(w)

X = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([len(z_data[0]), 25], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([25, 4], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([25]), name="Bias1")
b2 = tf.Variable(tf.zeros([4]), name="Bias2")

L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
hy = tf.sigmoid(tf.matmul(L2, W2) + b2)

saver = tf.train.Saver()

with tf.device('/gpu:0'):
    with tf.Session() as sess:
        saver.restore(sess, "/temp/model.ckpt")

        classification = (sess.run([hy], feed_dict={X: z_data})[0]).tolist()
        print(classification)
        i = 0
        accuracy = 0
        while i < len(classification):
            predicted_major = 0
            real_mayor = 0
            j = 0
            while j < 4:
                if classification[i][j] > classification[i][predicted_major]:
                    predicted_major = j
                if w_data[i][j] > w_data[i][real_mayor]:
                    real_mayor = j
                j += 1
            if real_mayor == predicted_major:
                accuracy += 1
            i += 1
        final_accuracy = float(accuracy)*100/float(i)
        print('{}%'.format(format(final_accuracy, '.2f')))
