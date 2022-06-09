# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn import tree
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import torch
import math
import random
import scipy.stats

def threshold_determination(Q):

    new_data = []
    for erer in Q:
        new_data=new_data+(np.ones(5)*erer).tolist()

    value_var = []
    Q_s=np.sort(new_data)
    for i in range(0,len(new_data)):
        L_l=Q_s[0:i]
        L_r = Q_s[i+1:len(Q_s)]
        value_var.append(abs(np.std(L_l) - np.std(L_r)))

    threshold_value=Q_s[np.where(value_var == np.min(value_var[1:len(value_var)-1]))]

    return threshold_value

def auditing_function(data,model=0):

    out=[]
    para = []
    q = 100
    for re in model.coef_[0]:
        para.append(re)

    for n in range(0, len(data)):
      out_temp = data[n] + q*2*(np.ones(1)-np.array(para).transpose().dot( data[n])) * np.array(para)
      out.append(out_temp.reshape(2))
    return np.array(out)

def eucliDist(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))


#setting
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.rc('font',family='Times New Roman')

#data import
data = pd.read_table('data.txt', header=None, delim_whitespace=True)  #.sample(frac=1)
print(data.info())
print(data.head())
auditing_data = np.array(data.loc[:][[0, 1]])
auditing_data_label = np.array(data[2])
auditing_data_label = np.where(auditing_data_label == 1, 1, -1)

train_data = auditing_data[0:50]
train_data_label = auditing_data_label[0:50]

non_train_data = auditing_data[51:100]
non_train_data_label = auditing_data[51:100]

target_model = SVC(kernel='linear', tol=1e-30, shrinking=False, probability=True, C=1, max_iter=20000)
target_model.fit(train_data, train_data_label)

testdata_aduiting_traing_label = target_model.predict(auditing_data)

target_model_simulation = SVC(kernel='linear', tol=1e-30, shrinking=False, probability=True, C=1, max_iter=2000000)
target_model_simulation.fit(auditing_data, testdata_aduiting_traing_label)




noise_data = auditing_function(auditing_data, model = target_model_simulation)

noise_data_output = target_model.predict_proba(noise_data)
auditing_data_output = target_model.predict_proba(auditing_data)

diff = []
for e in range(0, len(auditing_data_output)):
    diff.append(eucliDist(noise_data_output[e], auditing_data_output[e]))
diff_1 = np.array(diff)

group_pre = []
group_pre_lable = []
group_size=7
for j in range(0, 100, group_size):
    group_pre.append(np.average(diff_1[j:j + group_size]))
    group_pre_lable.append(np.average(auditing_data_label[j:j + group_size]))


threld = threshold_determination(group_pre)
#or
# threld_1 = group_pre[np.argsort(group_pre)[int(50/group_size)]]
connect_label = [1 if sample >= threld else 0 for sample in group_pre]
audtting_result = []
for toy in connect_label:
    # audtting_result.append((np.zeros(5)+toy).tolist())
    audtting_result = audtting_result + (np.zeros(group_size) + toy).tolist()
audtting_result=np.array(audtting_result[0:50])


#plot (a)
plt.scatter(x=auditing_data[auditing_data_label == 1, 0], y=auditing_data[auditing_data_label == 1, 1],
                             s=80, marker='o', color='#e9a3c9', zorder=20)
plt.scatter(x=auditing_data[auditing_data_label == -1, 0], y=auditing_data[auditing_data_label == -1, 1],
                             s=80, marker='x', color='#a1d76a', zorder=20)

plt.scatter(y=train_data[0:50,1],x=train_data[0:50,0], s=80, marker='*', color='black', zorder=20)

for i in range(0, 1):
    w = target_model.coef_[i]
    a = - w[0] / w[1]
    xx = np.linspace(-2.5, 3)
    yy = a * xx - (target_model.intercept_[i]) / w[1]
    # Plot the hyperplane
    plt.plot(xx, yy, color='black', linestyle="--")

plt.show()

#plot (b)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.rc('font', family='Times New Roman')

plt.scatter(x=auditing_data[auditing_data_label == 1, 0], y=auditing_data[auditing_data_label == 1, 1],
                             s=80, marker='o', color='#e9a3c9', zorder=20)
plt.scatter(x=auditing_data[auditing_data_label == -1, 0], y=auditing_data[auditing_data_label == -1, 1],
                             s=80, marker='x', color='#a1d76a', zorder=20)
plt.scatter(x=auditing_data[np.where(audtting_result == 1), 0], y=auditing_data[np.where(audtting_result == 1), 1],
            s=200, marker="o", color='', edgecolors='r')
plt.scatter(y=auditing_data[0:50, 1], x=auditing_data[0:50, 0], s=80, marker='*', color='black', zorder=20)

plt.show()
