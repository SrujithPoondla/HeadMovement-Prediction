# -*- coding:utf-8 -*-
import csv
import matplotlib.pyplot as plt
import numpy as np
import pyrenn as prn
import pandas as pd

x = pd.read_csv('train_input.txt',index_col=False)
t1=np.array(x['time'].values)
P = np.array([x['pitch'].values,x['roll'].values])
print   P
y = pd.read_csv('train_output.txt',index_col = False)
Y = np.array([y['pitch'].values,y['roll'].values])

x = pd.read_csv('test_output_100ms.txt',index_col=False)
Ptest= np.array([x['pitch'].values,x['roll'].values])
print   P
y = pd.read_csv('test_input_100ms.txt',index_col = False)
Ytest= np.array([y['pitch'].values,y['roll'].values])
t2=np.array(y['time'].values)


print 'Creating'
net = prn.CreateNN([2,7,7,2])
print 'Start training...'
prn.train_LM(P,Y,net, verbose=True, k_max=40, E_stop=1e-3)
###
##Calculate outputs of the trained NN for train and test data
y = prn.NNOut(P,net)
ytest = prn.NNOut(Ptest,net)
print ytest
###
#Plot results
fig = plt.figure(figsize=(15,10))
ax0 = fig.add_subplot(221)
ax1 = fig.add_subplot(222,sharey=ax0)
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(224,sharey=ax2)
fs=18
#t1 = np.arange(0,2227.0)/4 #480 timesteps in 15 Minute resolution
#t1= pd.read_csv('train_input.txt',index_col = False)
#t2 = np.arange(0,982.0)/4 #480 timesteps in 15 Minute resolution
#Train Data
print len(t1)
print len(y[0])
print len(Y[0])
print "---------"
print len(t2)
print len(ytest[0])
print len(Ytest[0])

ax0.set_title('Train Data',fontsize=fs)
ax0.plot(t1,y[0],color='b',lw=2,label='NN Output')
ax0.plot(t1,Y[0],color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Data')
ax0.tick_params(labelsize=fs-2)
ax0.legend(fontsize=fs-2,loc='upper left')
ax0.grid()
ax0.set_ylabel(' Pitch [Angle]',fontsize=fs)
plt.setp(ax0.get_xticklabels(), visible=False)

ax2.plot(t1,y[1],color='b',lw=2,label='NN Output')
ax2.plot(t1,Y[1],color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Train Data')
ax2.tick_params(labelsize=fs-2)
ax2.grid()
ax2.set_xlabel('Time [ms]',fontsize=fs)
ax2.set_ylabel('Roll [Angle]',fontsize=fs)





#Test Data
ax1.set_title('Test Data',fontsize=fs)
ax1.plot(t2,ytest[0],color='b',lw=2,label='NN Output')
ax1.plot(t2,Ytest[0],color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Test Data')
ax1.tick_params(labelsize=fs-2)
ax1.set_ylabel(' Pitch [Angle]',fontsize=fs)
ax1.legend(fontsize=fs-2,loc='upper left')
ax1.grid()
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax1.get_yticklabels(), visible=False)

ax3.plot(t2,ytest[1],color='b',lw=2,label='NN Output')
ax3.plot(t2,Ytest[1],color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Test Data')
ax3.tick_params(labelsize=fs-2)
ax3.grid()
ax3.set_xlabel('Time [ms]',fontsize=fs)
plt.setp(ax3.get_yticklabels(), visible=False)
ax3.set_ylabel('Roll [Angle]',fontsize=fs)

fig.tight_layout()
plt.show()
