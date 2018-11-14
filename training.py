import tensorflow as tf
import numpy as np
import csv

users = {}
x = []
with open('training_set.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile)
    itercsv = iter(spamreader)
    next(itercsv)
    for row in spamreader:
        users[row[0]] = None
        x.append([float(i) for i in row[2:]])
x_data = np.array(x)

y = []
with open('major_index.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile)
    itercsv = iter(spamreader)
    next(itercsv)
    for row in spamreader:
        user = row[0]
        if user in users:
            y.append([float(j) for j in row[2:]])
y_data = np.array(y)

learning_rate = 0.001
training_epochs = 32
n_input = len(x[0])
n_hidden = 10
n_output = 4

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([n_hidden]), name="Bias1")
b2 = tf.Variable(tf.zeros([n_output]), name="Bias2")

L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
hy = tf.sigmoid(tf.matmul(L2, W2) + b2)

cost = tf.reduce_mean(-Y*tf.log(hy) - (1-Y)*tf.log(1-hy))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sesh:
    sesh.run(init)

    for step in range(training_epochs):
        sesh.run(optimizer, feed_dict={X: x_data, Y: y_data})

        if step % 1000 == 0:
            print(sesh.run(cost, feed_dict={X: x_data, Y: y_data}))

    answer = tf.equal(tf.floor(hy + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(answer, "float"))

    print(sesh.run([hy], feed_dict={X: x_data, Y: y_data}))
    print("Accuracy: ", accuracy.eval({X: x_data, Y: y_data}))
