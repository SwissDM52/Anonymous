# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVC
import math
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import torchvision
from sklearn import metrics


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

        if self.transform is not None:
            # 转换成PIL
            pil_image = Image.fromarray(np.uint8(img))
            img = self.transform(pil_image)

        return img,target

    def __len__(self):
        return len(self.images)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, 2)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def auditing_function(data,model=0):

    out=[]
    para = []
    q = 1000
    for re in model.coef_[0]:
        para.append(re)

    for n in range(0, len(data)):
      out_temp = data[n] + q*2*(np.ones(1)-np.array(para).transpose().dot( data[n])) * np.array(para)
      out.append(out_temp.reshape(len(data[n])))
    return np.array(out)


def eucliDist(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))


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


transform = transforms.Compose([Resize((32, 32)),
    # transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    #                             transforms.RandomHorizontalFlip(),
                                transforms.Grayscale (num_output_channels=1) ,
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])


#Note: because of copyright, please you can download vggdata from https://www.robots.ox.ac.uk/~vgg/data/vgg_face/.
#[1] O. M. Parkhi, A. Vedaldi, A. Zisserman. Deep Face Recognition. British Machine Vision Conference, 2015
data_all = torchvision.datasets.ImageFolder(root='data\\vggdata',transform=transform)
data_all_test = torchvision.datasets.ImageFolder(root='data\\auditingdata',transform=transform)
data_all_test_o = torchvision.datasets.ImageFolder(root='data\\testdata',transform=transform)

data_loader_train = torch.utils.data.DataLoader(dataset=data_all,
                                                batch_size = 32,
                                                shuffle = True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_all_test,
                                                batch_size = 1,
                                                shuffle = False)

data_loader_test_o = torch.utils.data.DataLoader(dataset=data_all_test_o,
                                                batch_size = 1,
                                                shuffle = False)

net_args = {
    "block": ResidualBlock,
    "layers": [2, 2, 2, 2]
}

model = ResNet(**net_args).cuda()

# print (net)
criterion=nn.CrossEntropyLoss().cuda()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.9)


total_epoches = 1000 # 50
step_size = 10     # 10
base_lr = 0.01    # 0.01

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
print(model)
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()


for epoch in range(total_epoches):
    # trainning
    ave_loss = 0
    for batch_idx, (xx, yy) in enumerate(data_loader_train):
        if torch.cuda.is_available():
            xx = xx.cuda()
            yy = yy.cuda()

        x, target = Variable(xx), Variable(yy)
        optimizer.zero_grad()
        out = model(x.float())
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()


        if (batch_idx + 1) % 32 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(x), len(data_loader_train.dataset),
                       100. * (batch_idx + 1) / len(data_loader_train), loss.data.item()))

    # torch.save(model.state_dict(), "model_parameter.pkl")
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
torch.save(model.state_dict(), "model_parameter_1000.pkl")
# model.load_state_dict(torch.load ( "model_parameter_1000.pkl" ))

correct=0
total=0

for batch_idx, (xx, yy) in enumerate(data_loader_test_o):
            if torch.cuda.is_available():
                xx = xx.cuda()
                yy = yy.cuda()
            x, target = Variable(xx), Variable(yy)
            out = model(x.float())
            m = nn.Softmax(dim=1)
            _, predicted = torch.max(out, 1)

            out1 = m(out).tolist()  # m(out)
            total += target.size(0)
            correct+=(predicted.item()==target).sum()
print('Accuracy of the network on the %d test images: %d %%' % (total , 100 * correct / total))


####Test####
DNN_results=[]
DNN_results_label=[]
auditing_data_label_1=[]
correct=0
total=0
test_x=[]
for batch_idx, (xx, yy) in enumerate(data_loader_test):
            if torch.cuda.is_available():
                xx = xx.cuda()
                yy = yy.cuda()
            x, target = Variable(xx), Variable(yy)

            test_x.append(x.cpu().detach().numpy().reshape(1024).tolist())
            auditing_data_label_1.append(target.cpu().detach().numpy().tolist()[0])
            out = model(x.float())
            # DNN_temp_pre.append(out.tolist())
            m = nn.Softmax(dim=1)
            _, predicted = torch.max(out, 1)

            out1 = m(out).tolist()  # m(out)
            DNN_results_label.append(predicted.item())
            DNN_results.append(out1[0])
            total += target.size(0)
            correct+=(predicted.item()==target).sum()
print('Accuracy of the network on the %d test images: %d %%' % (total , 100 * correct / total))


target_model_simulation = SVC(kernel='linear', tol=1e-30, shrinking=False, probability=True, C=1, max_iter=20000)
target_model_simulation.fit(test_x, DNN_results_label)

noise_data = auditing_function(test_x, model = target_model_simulation)

X_test_noise_tensor=[]
y_test_noise_tensor=[]
for dd in range(0,len(noise_data)):

   X_test_noise_tensor.append(torch.tensor(noise_data[dd], dtype=torch.double).reshape(1, 1024))
   y_test_noise_tensor.append(torch.tensor(DNN_results_label, dtype=torch.float))

x_test_noise_pyt = trainset(xx=X_test_noise_tensor,yy=y_test_noise_tensor, transform=transform)
data_loader_noise_test = torch.utils.data.DataLoader(dataset=x_test_noise_pyt,
                                                batch_size = 1,
                                                shuffle = False)

DNN_results_noise=[]
DNN_results_label_noise=[]
for batch_idx, (xx, yy) in enumerate(data_loader_noise_test):
            if torch.cuda.is_available():
                xx = xx.cuda()
                yy = yy.cuda()
            x, target = Variable(xx), Variable(yy)
            out = model(x.float())
            m = nn.Softmax(dim=1)
            _, predicted = torch.max(out, 1)

            out1 = m(out).tolist()  # m(out)
            DNN_results_label_noise.append(predicted.item())
            DNN_results_noise.append(out1[0])

diff = []
for e in range(0, len(DNN_results_noise)):
    diff.append(eucliDist(DNN_results_noise[e], DNN_results[e]))
    # diff.append(scipy.stats.entropy(DNN_results_noise[e], DNN_results[e]))
diff_1 = np.array(diff)


group_pre = []
group_pre_lable = []
group_pre_lable_1 = []
group_size=5
for j in range(0, len(diff_1), group_size):
    group_pre.append(np.average(diff_1[j:j + group_size]))
    group_pre_lable.append(np.average(auditing_data_label_1[j:j + group_size]))
    group_pre_lable_1.append(np.average(auditing_data_label_1[j:j + group_size]))
threld = threshold_determination(group_pre)
#or

fpr, tpr, thresholds = metrics.roc_curve(group_pre_lable_1, group_pre)
AUC_result = metrics.auc(fpr, tpr)
# Acc_result = metrics.accuracy_score(group_pre_lable_1, group_pre)
# F_result = metrics.f1_score(group_pre_lable_1, group_pre)
print("AUC_result:", AUC_result)
# print("Acc_result:", Acc_result)
# print("F_result:", F_result)
