# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
import torch
import math

from torch.autograd import Variable


import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset


def threshold_determination(Q):

    new_data = []
    for erer in Q:
        new_data=new_data+(np.ones(5)*erer).tolist()

    value_var = []
    Q_s=np.sort(new_data)
    for i in range(0,len(new_data)):
        L_l=Q_s[0:i]
        L_r = Q_s[i+1:len(Q_s)]
        value_var.append(abs(np.var(L_l) - np.var(L_r)))

    threshold_value=Q_s[np.where(value_var == np.min(value_var[1:len(value_var)-1]))]

    return threshold_value

def auditing_function(data,model=0):

    out=[]
    para = []
    q = 10000
    for re in model.coef_[0]:
        para.append(re)

    for n in range(0, len(data)):
      out_temp = data[n] + q*2*(np.ones(1)-np.array(para).transpose().dot( data[n])) * np.array(para)
      out.append(out_temp.reshape(2))
    return np.array(out)


def eucliDist(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))

class trainset():
    def __init__(self, xx=None,yy=None,transform=None):
        #定义好 image 的路径
        self.images = xx
        self.target = yy
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        # img = self.loader(fn)
        target = self.target[index]

        # if self.transform is not None:
        #     # 转换成PIL
        #     pil_image = Image.fromarray(np.uint8(img))
        #     img = self.transform(pil_image)

        return img,target

    def __len__(self):
        return len(self.images)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.connected_layer =  torch.nn.Sequential(torch.nn.Linear(2, 64))

        self.connected_layer2 =  torch.nn.Sequential(torch.nn.Linear(64, 2))

    def forward(self, x):
        x = self.connected_layer(x)
        x = self.connected_layer2(x)
        x = x.view(-1, 2)
        # x = self.dense(x)
        return x


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
auditing_data_label = np.where(auditing_data_label == 1, 1, 0)

train_data = auditing_data[0:50]
train_data_label = auditing_data_label[0:50]

non_train_data = auditing_data[51:100]
non_train_data_label = auditing_data[51:100]

X_train_tensor=[]
y_train_tensor=[]

for dd in range(0,50):

   X_train_tensor.append(torch.tensor(train_data[dd], dtype=torch.double).reshape(-1,1, 2))
   y_train_tensor.append(torch.tensor(auditing_data_label[dd], dtype=torch.long))


x_train_pyt = trainset(xx=X_train_tensor,yy=y_train_tensor)
data_loader_train = torch.utils.data.DataLoader(dataset=x_train_pyt,
                                                batch_size = 8,
                                                shuffle = True)

X_test_tensor=[]
y_test_tensor=[]
for dd in range(0,len(auditing_data)):

   X_test_tensor.append(torch.tensor(auditing_data[dd], dtype=torch.double).reshape(1, 2))
   y_test_tensor.append(torch.tensor(auditing_data_label[dd], dtype=torch.float))


x_test_pyt = trainset(xx=X_test_tensor,yy=y_test_tensor)

data_loader_test = torch.utils.data.DataLoader(dataset=x_test_pyt,
                                                batch_size = 1,
                                                shuffle = False)

total_epoches = 200 # 50
step_size = 10     # 10
base_lr = 0.01    # 0.01
target_model = Model()
optimizer = optim.SGD(target_model.parameters(), lr=base_lr)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

print(target_model)
if torch.cuda.is_available():
    target_model = target_model.cuda()
    criterion = criterion.cuda()

####training target model
for epoch in range(total_epoches):
    # trainning
    ave_loss = 0
    for batch_idx, (xx, yy) in enumerate(data_loader_train):
        if torch.cuda.is_available():
            xx = xx.cuda()
            yy = yy.cuda()

        x, target = Variable(xx), Variable(yy)
        optimizer.zero_grad()
        out = target_model(x.float())
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(x), len(data_loader_train.dataset),
                       100. * (batch_idx + 1) / len(data_loader_train), loss.data.item()))

    # torch.save(model.state_dict(), "model_parameter.pkl")
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    # exp_lr_scheduler.step()

DNN_temp_pre = []
grad_nosie = []

testdata_aduiting_traing_label=[]
auditing_data_output=[]
for batch_idx, (xx, yy) in enumerate(data_loader_test):
    if torch.cuda.is_available():
                xx = xx.cuda()
                yy = yy.cuda()

    x, target = Variable(xx), Variable(yy)
    out = target_model(x.float())
    _, predicted = torch.max(out, 1)
    testdata_aduiting_traing_label.append(predicted.tolist())
    m = nn.Softmax(dim=1)
    out1 = m(out)
    out1 = out1.tolist()
    auditing_data_output = auditing_data_output+out1


target_model_simulation = SVC(kernel='linear', tol=1e-30, shrinking=False, probability=True, C=1, max_iter=2000000)
target_model_simulation.fit(auditing_data, testdata_aduiting_traing_label)


noise_data = auditing_function(auditing_data, model = target_model_simulation)
X_test_tensor_noise=[]
y_test_tensor_noise=[]
for dd in range(0, len(noise_data)):
    X_test_tensor_noise.append(torch.tensor(noise_data[dd], dtype=torch.double).reshape(-1, 1, 2))
    y_test_tensor_noise.append(torch.tensor(testdata_aduiting_traing_label[dd], dtype=torch.float))

x_nosie_pyt = trainset(xx=X_test_tensor_noise, yy=y_test_tensor_noise)

data_loader_test_noise = torch.utils.data.DataLoader(dataset=x_nosie_pyt,
                                                       batch_size=1,
                                                       shuffle=False)
noise_data_output = []
for batch_idx, (xx, yy) in enumerate(data_loader_test_noise):
        if torch.cuda.is_available():
                xx = xx.cuda()
                yy = yy.cuda()

        x, target = Variable(xx), Variable(yy)
        out = target_model(x.float())
            # DNN_temp_pre.append(out.tolist())
        m = nn.Softmax(dim=1)
        out1 = m(out)  # m(out)
        out1 = out1.tolist()
        noise_data_output.append(out1[0])

diff = []
for e in range(0, len(auditing_data_output)):
    diff.append(eucliDist(noise_data_output[e], auditing_data_output[e]))
diff_1 = np.array(diff)

group_pre = []
group_pre_lable = []
group_size=5
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
plt.scatter(x=auditing_data[auditing_data_label == 0, 0], y=auditing_data[auditing_data_label == 0, 1],
                             s=80, marker='x', color='#a1d76a', zorder=20)

plt.scatter(y=train_data[0:50,1],x=train_data[0:50,0], s=80, marker='*', color='black', zorder=20)

for parameters in target_model.parameters():
    print(parameters)
    grad_nosie.append(parameters.data.tolist())

w1 = np.array(grad_nosie[0])
b1 = np.array(grad_nosie[1])
w2 = np.array(grad_nosie[2])
b2 = np.array(grad_nosie[3])

w_mix = np.dot(w2, w1)
b_mix = np.dot(w2, b1) + b2
delta_w = np.dot(b1, 1 / w1)
delta_w = np.dot(b_mix, 1 / w_mix.T)
for i in range(0, 1):
    w =  w_mix[i]
    a = - w[0]/ w[1]
    xx = np.linspace(-2.5, 3)
    yy = a * xx - (b_mix[i]) / w[1]
    # Plot the hyperplane
    plt.plot(xx, yy, color='black')

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.rc('font', family='Times New Roman')
plt.show()

#plot (b)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.rc('font', family='Times New Roman')

plt.scatter(x=auditing_data[auditing_data_label == 1, 0], y=auditing_data[auditing_data_label == 1, 1],
                             s=80, marker='o', color='#e9a3c9', zorder=20)
plt.scatter(x=auditing_data[auditing_data_label == 0, 0], y=auditing_data[auditing_data_label == 0, 1],
                             s=80, marker='x', color='#a1d76a', zorder=20)
plt.scatter(x=auditing_data[np.where(audtting_result == 1), 0], y=auditing_data[np.where(audtting_result == 1), 1],
            s=200, marker="o", color='', edgecolors='r')
plt.scatter(y=auditing_data[0:50, 1], x=auditing_data[0:50, 0], s=80, marker='*', color='black', zorder=20)

plt.show()
