# -*- coding:utf-8 -*-
import csv
# import matplotlib.pyplot as plt
import numpy as np
import pyrenn as prn


x = []
y = []
with open('model071301.rnn', 'rb') as f:
	reader = csv.reader(f)
	print reader.data[1]
#    reader = csv.reader(f)
#    for row in reader:
#        if row == []:
#            continue
#        x.append(row[0])
#        #y.append(row[1])
#
#num_test = len(x) / 3
#x_test =  np.array(x[:num_test], np.float64)
#y_test =  np.array(y[:num_test], np.float64)
#x_train = np.array(x[num_test:], np.float64)
#y_train = np.array(y[num_test:], np.float64)
#
#print 'Creating'
#net = prn.CreateNN([1,3,3,1])
#print 'Start training...'
#prn.train_LM(x_train, y_train, net, verbose=True, k_max=200, E_stop=1e-3)
