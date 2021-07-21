from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
import torch.nn.functional as F
from manual import GetLoader
from torch.utils.data import DataLoader
learing_rate = 5*1e-5
epochs = 200
device ='cpu'
dim_hidden = 512
split_ratio = 9
bs = 32
class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.layer_hidden1 = nn.Linear(dim_hidden, dim_hidden)
        self.layer_hidden2 = nn.Linear(dim_hidden, dim_hidden)
        self.layer_hidden3 = nn.Linear(dim_hidden,  dim_out)
        self.layer_output = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer_input(x)
        x = self.relu(self.dropout(x))
        x = self.layer_hidden1(x)
        x = self.relu(self.dropout(x))
        x = self.layer_hidden2(x)
        x = self.relu(self.dropout(x))
        x = self.layer_output(x)
        return x

def train(data, label, global_model, device):
    optimizer = torch.optim.Adam(global_model.parameters(), lr=learing_rate)
    test_dataset = data[:slice//split_ratio, :]
    testlabel = label[:slice//split_ratio, :]
    train_dataset = data[slice//split_ratio:, :]
    trainlabel = label[slice//split_ratio:, :]
    torch_data = GetLoader(train_dataset, trainlabel)
    trainloader = DataLoader(torch_data, batch_size=bs, shuffle=False, drop_last=False)
    torch_data1 = GetLoader(test_dataset, testlabel)
    testloader = DataLoader(torch_data1, batch_size=bs, shuffle=False, drop_last=False)
    criterion = nn.MSELoss().to(device)
    epoch_loss = []
    iflist = []
    for epoch in range(epochs):
        batch_loss = []
        if_loss = []
        for i, data in enumerate(trainloader):
            images, labels = data[0].float().to(device), data[1].to(device).float()
            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('Train loss:', loss_avg, epoch)
        epoch_loss.append(loss_avg)
        net = global_model.eval()
        total, correct = 0.0, 0.0
        for t, data1 in enumerate(testloader):
            images1, labels1 = data1[0].float().to(device), data1[1].to(device).float()
            outputs1 = net(images1)
            testloss = criterion(outputs1, labels1)
            if_loss.append(testloss.item())
        test_loss_avg = sum(if_loss)/len(if_loss)
        print('Validation:', test_loss_avg)
        iflist.append(test_loss_avg)
        plt.figure()
        plt.plot(range(len(epoch_loss)), epoch_loss)
        plt.xlabel('epochs')
        plt.ylabel('Train Loss')
        plt.savefig('modelresult/loss.png')
        # testing
        plt.figure()
        plt.plot(range(len(iflist)), iflist)
        plt.xlabel('epochs')
        plt.ylabel('Test Loss')
        plt.savefig('modelresult/test.png')
    return epoch_loss, iflist, global_model.state_dict()


alldata1 = np.loadtxt('data/datak2.txt')
alldata2 = np.loadtxt('data/datak3.txt')
alldata3 = np.loadtxt('data/datak4.txt')
alldata4 = np.loadtxt('data/datak5.txt')
alldata5 = np.loadtxt('data/datak6.txt')
alldata = np.vstack((alldata1, alldata2, alldata3, alldata4, alldata5))
slice = alldata.shape[0]
traindata = alldata[:, :9]
label = alldata[:, 9:]
print('datasize', slice)
print(np.std(traindata, axis=0))
print(np.mean(traindata, axis=0))
exit()
traindata = (traindata-np.mean(traindata, axis=0))/np.expand_dims(np.std(traindata, axis=0),axis=0)
traindata = np.nan_to_num(traindata)
label = (label-np.mean(label, axis=0))/np.expand_dims(np.std(label, axis=0),axis=0)
permutation = np.random.permutation(label.shape[0])
shuffled_dataset = traindata[permutation, :]
shuffled_labels = label[permutation, :]
global_model = MLP(dim_in=traindata.shape[1], dim_hidden=dim_hidden, dim_out=label.shape[1])
global_model.to(device)
global_model.train()
print(global_model)
device ='cpu'
epoch_loss, iflist, modelparamater = train(shuffled_dataset, shuffled_labels, global_model, device)
torch.save(modelparamater,'modelresult/NNpk8s')



