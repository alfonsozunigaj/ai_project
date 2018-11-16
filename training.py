import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
import datetime

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

rec, rec_op = tf.metrics.recall(labels=X, predictions=Y)
acc, acc_op = tf.metrics.accuracy(labels=X, predictions=Y)
pre, pre_op = tf.metrics.precision(labels=X, predictions=Y)
auc, auc_op = tf.metrics.auc(labels=X, predictions=Y)

W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([n_hidden]), name="Bias1")
b2 = tf.Variable(tf.zeros([n_output]), name="Bias2")

L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
hy = tf.sigmoid(tf.matmul(L2, W2) + b2)

cost = tf.reduce_mean(-Y * tf.log(hy) - (1 - Y) * tf.log(1 - hy))

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

learning_rates = [0.1, 0.01, 0.001, 0.0001]
training_epochses = [1000, 10000, 15000, 60000]
for lr in learning_rates:
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    for te in training_epochses:
        print('NEW EPOCH')
        accuracies = []
        recalls = []
        precisions = []
        F1_scores = []
        AUCs = []
        with tf.device('/gpu:0'):
            with tf.Session() as sesh:
                sesh.run(init)
                start = datetime.datetime.now()
                for step in range(te):
                    sesh.run(optimizer, feed_dict={X: x_data, Y: y_data})

                    data = sesh.run(cost, feed_dict={X: x_data, Y: y_data})
                    if data is None:
                        continue
                    if step % 50 == 0:
                        outputs = (sesh.run([hy], feed_dict={X: z_data, Y: w_data})[0]).tolist()
                        predictions = []
                        for item in outputs:
                            max_index = item.index(max(item))
                            aux_list = [0, 0, 0, 0]
                            aux_list[max_index] = 1
                            predictions.append(aux_list)
                        predictions = np.array(predictions)
                        accuracy = sesh.run(acc_op, feed_dict={X: w_data, Y: predictions})
                        recall = sesh.run(rec_op, feed_dict={X: w_data, Y: predictions})
                        precision = sesh.run(pre_op, feed_dict={X: w_data, Y: predictions})
                        auc = sesh.run(auc_op, feed_dict={X: w_data, Y: predictions})
                        F1_score = 2*precision*recall/(precision+recall)
                        accuracies.append(accuracy)
                        recalls.append(recall)
                        precisions.append(precision)
                        AUCs.append(auc)
                        F1_scores.append(F1_score)
                        print(accuracy, recall, precision, F1_score, auc)
                end = datetime.datetime.now()
                print("Learning rate: {} and training epoch: {}".format(lr, te))
                print('\tSTART: {}\n\tEND: {}\n\tDURATION: {} seconds'.format(start.time(), end.time(), (end-start).seconds))
                print('\t#################\n\tAccuracy: {}\n\tRecall: {}\n\tPrecision: {}\n\tF1 Score: {}\n\tAUC: {}'.format(accuracies[-1], recalls[-1], precisions[-1], F1_scores[-1], AUCs[-1]))
                plt.figure(figsize=(16, 9), dpi=100)
                plt.plot(range(te//50), accuracies)
                plt.plot(range(te//50), recalls)
                plt.plot(range(te//50), precisions)
                plt.plot(range(te//50), F1_scores)
                plt.plot(range(te//50), AUCs)
                plt.axis([0, te//50, 0, 1])
                plt.xlabel('Epoch')
                plt.ylabel('Values')
                plt.legend(['Accuracy', 'Recall', 'Precision', 'F1 Score', 'AUC'])
                plt.title('Learning_rate: {} and Training_epochs: {}'.format(lr, te))
                plt.axis()
                plt.savefig("lr{}_te{}_training.png".format(lr, te))
                plt.show()
