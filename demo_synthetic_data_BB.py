# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
import math
from sklearn import metrics


def threshold_determination(Q):

    new_data = []
    for q in Q:
        new_data = new_data+(np.ones(5)*q).tolist()

    value_var = []
    Q_s = np.sort(new_data)

    for i in range(0, len(Q_s)):
        L_l = Q_s[0:i]
        L_r = Q_s[i+1:len(Q_s)]
        value_var.append(abs(np.var(L_l) - np.var(L_r)))

    index_t = np.where(value_var == np.min(value_var[1:len(value_var) - 1]))[0][0]
    while Q_s[index_t] == Q_s[index_t-1]:
        index_t=index_t-1
    threshold_value = Q_s[index_t-1]
    return threshold_value

def auditing_function(data,model=0):

    out = []
    para = []
    q = 0.01
    for re in model.coef_[0]:
        para.append(re)

    for n in range(0, len(data)):
      out_temp = data[n] + q*2*(np.ones(1)-np.array(para).transpose().dot(data[n])) * np.array(para)
      out.append(out_temp.reshape(2))
    return np.array(out)

def eucliDist(A,B):
    return math.sqrt((A-B)**2)

#setting
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.rc('font',family='Times New Roman')
result_label = 1 #which label we would like audit?
group_size = 10


#data import
data = pd.read_table('data.txt', header=None, delim_whitespace=True,)
print(data.info())
print(data.head())
auditing_data = np.array(data.loc[:][[0, 1]])
auditing_data_label = np.array(data[2])
auditing_data_label = np.where(auditing_data_label == 1, 1, -1)

train_data = auditing_data[0:50]
train_data_label = auditing_data_label[0:50]
train_data_index = train_data_label.argsort()
train_data = train_data[train_data_index]
train_data_label = train_data_label[train_data_index]


non_train_data = auditing_data[50:100]
non_train_data_label = auditing_data_label[50:100]
non_train_data_index = non_train_data_label.argsort()
non_train_data = non_train_data[non_train_data_index]
non_train_data_label = non_train_data_label[non_train_data_index]

#target model
target_model = SVC(kernel='linear', tol=1e-30, shrinking=False, probability=True, C=0.01, max_iter=20000)
target_model.fit(train_data, train_data_label)

#auditing data preparation
auditing_data = np.vstack((train_data,non_train_data))
auditing_data_label = np.hstack((train_data_label,non_train_data_label))
auditing_data_train_non_train_label = np.array([1] * len(train_data) +[-1] * len(non_train_data))

#target model simulation

auditing_data_prediction_label = target_model.predict(auditing_data)

target_model_simulation = SVC(kernel='linear', tol=1e-30, shrinking=False, probability=True, C=0.01, max_iter=20000)
target_model_simulation.fit(auditing_data, auditing_data_prediction_label)

#auditing data processed by auditing function
noise_data = auditing_function(auditing_data, model = target_model_simulation)

# differential value
noise_data_output = target_model.predict_proba(noise_data)
auditing_data_output = target_model.predict_proba(auditing_data)
diff = []
for e in range(0, len(auditing_data_output)):
    diff.append(eucliDist(noise_data_output[e][0], auditing_data_output[e][0]))
diff_1 = np.array(diff)


# group data generation
index_rl = np.where(auditing_data_label == result_label)
diff_2 = diff_1[index_rl]
auditing_data_train_nontrain_label_1 = auditing_data_train_non_train_label[index_rl]

group_prediction = []
group_true_lable = []


for j in range(0, len(index_rl[0]), group_size):
    if len(diff_1[index_rl[0][j:j + group_size]]) == group_size:

       if np.average(auditing_data_train_non_train_label[index_rl[0][j:j + group_size]]) == 1:
            group_true_lable.append(1)
            group_prediction.append(np.average(diff_1[index_rl[0][j:j + group_size]]))
       elif np.average(auditing_data_train_non_train_label[index_rl[0][j:j + group_size]]) == -1:
           group_true_lable.append(0)
           group_prediction.append(np.average(diff_1[index_rl[0][j:j + group_size]]))


threld = threshold_determination(group_prediction)
connect_label = [1 if sample > threld else 0 for sample in group_prediction]

fpr, tpr, thresholds = metrics.roc_curve(group_true_lable, connect_label)
AUC_result = metrics.auc(fpr, tpr)
F_result = metrics.f1_score(group_true_lable, connect_label)

print("AUC_result:", AUC_result)
print("F_result:", F_result)
