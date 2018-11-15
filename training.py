import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt

users_training = {}
x = []
with open('training_set.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile)
    itercsv = iter(spamreader)
    next(itercsv)
    for row in spamreader:
        users_training[row[0]] = None
        x.append([float(i) for i in row[2:]])
x_data = np.array(x)

users_testing = {}
z = []
with open('testing_set.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile)
    itercsv = iter(spamreader)
    next(itercsv)
    for row in spamreader:
        users_testing[row[0]] = None
        z.append([float(i) for i in row[2:]])
z_data = np.array(z)

y = []
w = []
with open('major_index.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile)
    itercsv = iter(spamreader)
    next(itercsv)
    for row in spamreader:
        user = row[0]
        if user in users_training:
            y.append([float(j) for j in row[2:]])
        elif user in users_testing:
            w.append([float(j) for j in row[2:]])
y_data = np.array(y)
w_data = np.array(w)

learning_rate = 0.01
training_epochs = 15000
n_input = len(x[0])
n_hidden = 15
n_output = 4

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([n_hidden]), name="Bias1")
b2 = tf.Variable(tf.zeros([n_output]), name="Bias2")

L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
hy = tf.sigmoid(tf.matmul(L2, W2) + b2)

cost = tf.reduce_mean(-Y * tf.log(hy) - (1 - Y) * tf.log(1 - hy))

init = tf.initialize_all_variables()

learning_rates = [0.1, 0.01, 0.001, 0.0001]
training_epochses = [1000, 10000, 15000, 60000]
for lr in learning_rates[1:]:
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    for te in training_epochses:
        with tf.device('/gpu:0'):
            with tf.Session() as sesh:
                sesh.run(init)
                costs = []
                counter = 0
                for step in range(te):
                    sesh.run(optimizer, feed_dict={X: x_data, Y: y_data})

                    data = sesh.run(cost, feed_dict={X: x_data, Y: y_data})
                    if data is None:
                        continue
                    costs.append(data)
                    counter += 1
                    if step % 1000 == 0:
                        print(sesh.run(cost, feed_dict={X: x_data, Y: y_data}))

                answer = tf.equal(tf.floor(hy + 0.5), Y)
                accuracy = tf.reduce_mean(tf.cast(answer, "float"))

                outputs = sesh.run([hy], feed_dict={X: z_data, Y: w_data})
                print(outputs)
                result = accuracy.eval({X: z_data, Y: w_data})
                print("Accuracy: ", result)

                fig, ax = plt.subplots()
                ax.plot(range(counter), costs)
                ax.axis([0, te, 0, 1])
                ax.set(xlabel='Training epochs', ylabel='Cost',
                       title='Accuracy: {}'.format(result))
                ax.grid()
                fig.savefig("lr{}_te{}_training.png".format(lr, te))

                plt.show()
