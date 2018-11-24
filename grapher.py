import numpy as np
from matplotlib import pyplot as plt


my_data = np.load('data.npy').item()
for key in dict(my_data):
    print('Learning rate: ', key)
    for k in my_data[key]:
        print('\tTraining epochs: ', k)
        print('\t\tAccuracy: ', my_data[key][k]['accuracy'])
        print('\t\tRecall: ', my_data[key][k]['recall'])
        print('\t\tAUC: ', my_data[key][k]['auc'])
