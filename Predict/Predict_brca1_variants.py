# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from scipy import interp
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


KF = KFold(n_splits = 5)
tprs=[]
aucs=[]
Accuracy = []
Recall = []
Precision = []
F1score = []
Average_precision = []
mean_fpr=np.linspace(0,1,100)
k = 0
precs=[]
aucs2=[]
mean_recall=np.linspace(0,1,100)

seq_stop = pd.read_csv('./brct_stop_BERT.csv',dtype=float)
seq = pd.read_csv('./brct_BERT.csv',dtype=float)
data = pd.read_csv('./BRCT_output.csv',dtype=str)
stop = pd.read_csv('./BRCT_stop.csv',dtype=str)

d = np.array(data)
s = np.array(seq)
ss = np.array(seq_stop)
stop = np.array(stop)

Hidden_size = 128
sample_len = d.shape[0]
time_len = d.shape[1] -1
embed_win = s.shape[1]-1
stop_len = ss.shape[0]
Batch_size = 64

x_bert = s[:, 0:embed_win]
x_ss = ss[:, 0:embed_win]
y_ss = ss[:, embed_win]
x = d[:, 0:time_len]
y = d[:, time_len]
x_stop = stop[:, 0:time_len]
y_stop = stop[:, time_len]

layer = 256+embed_win

#Hydrophilic coding
vocab1 = {'A': 1.8, 'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H' :-3.2,'I':4.5,
         'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2,'*':-100}
#Benign:0; Pathogenic:1
tags = {'0':0, '1':1}

y = [[tags[word] for word in sentence] for sentence in y]
x1 = [[vocab1[word] for word in sentence] for sentence in x]
x1 = np.array(x1).reshape(sample_len,time_len)
y = np.array(y).reshape(sample_len,1)
x1 = np.pad(x1,((0,0),(0,256-time_len)),'constant')

y_stop = [[tags[word] for word in sentence] for sentence in y_stop]
x_stop = [[vocab1[word] for word in sentence] for sentence in x_stop]
x_stop = np.array(x_stop).reshape(stop_len,time_len)
y_stop = np.array(y_stop).reshape(stop_len,1)
x_stop = np.pad(x_stop,((0,0),(0,256-time_len)),'constant')


X = np.concatenate((x_bert,x1),axis=1)
X_stop = np.concatenate((x_ss,x_stop),axis=1)
X = np.concatenate((X,X_stop),axis=0)
y = np.concatenate((y,y_stop),axis=0)

index1 = np.argwhere(y==1)
y_p = y[index1]
X_p = X[index1]
index2 = np.argwhere(y==0)
y_n = y[index2]
X_n = X[index2]

for train_index1,test_index1 in KF.split(X_p):
    for train_index2, test_index2 in KF.split(X_n):
        X_p_train, X_p_test = X_p[train_index1], X_p[test_index1]
        Y_p_train, Y_p_test = y_p[train_index1], y_p[test_index1]

        X_n_train, X_n_test = X_n[train_index2], X_n[test_index2]
        Y_n_train, Y_n_test = y_n[train_index2], y_n[test_index2]

        X_train = np.concatenate((X_p_train,X_n_train),axis=0)
        X_test = np.concatenate((X_p_test,X_n_test),axis=0)
        Y_train = np.concatenate((Y_p_train,Y_n_train),axis=0)
        Y_test = np.concatenate((Y_p_test,Y_n_test),axis=0)

        class Classifier(nn.Module):
            def __init__(self):
                super(Classifier, self).__init__()
                self.out = nn.Sequential(
                    nn.Linear(layer, 128),  # 输出层
                    nn.ReLU(),
                    nn.Linear(128, 2),
                    # nn.Softmax(dim=1)

                )

            def forward(self, x):
                x = x.view(x.size(0), -1)
                output = self.out(x)

                return output  # return x for visualization


        classi = Classifier()
        classi.cuda()
        print('Classifier:')
        print(classi)

        optimizer2 = torch.optim.Adam(classi.parameters(), lr=0.0001, weight_decay=0.005)  # optimize all parameters
        loss_func2 = nn.CrossEntropyLoss()
        Data = X_train
        Label = Y_train

        # 创建子类
        class subDataset(Dataset.Dataset):
            # 初始化，定义数据内容和标签
            def __init__(self, Data, Label):
                self.Data = Data
                self.Label = Label

            # 返回数据集大小
            def __len__(self):
                return len(self.Data)

            # 得到数据内容和标签
            def __getitem__(self, index):
                data = torch.LongTensor(self.Data[index])
                label = torch.LongTensor(self.Label[index])

                return data, label

        if __name__ == '__main__':
            dataset = subDataset(Data, Label)

            dataloader = DataLoader.DataLoader(dataset, batch_size=Batch_size, shuffle=True, num_workers=4)

        # training and testing
        for epoch in range(80):
            for i, (data, label) in enumerate(dataloader):
                data = data.cuda()
                label = label.cuda()
                b_x = data.view(-1,layer,1)  # reshape x to (batch, time_step, input_size)
                b_x = b_x.float()
                out = classi(b_x)
                label = label.flatten()
                out = torch.squeeze(out)

                loss = loss_func2(out, label)   # cross entropy loss
                optimizer2.zero_grad()           # clear gradients for this training step
                loss.backward()            # backpropagation, compute gradients
                optimizer2.step()                # apply gradients
                if i % 50 ==0:
                    print('Epoch:',epoch,'|train loss %.4f' %loss.cpu().data.numpy())

        testData = X_test
        testLabel = Y_test

        # 创建子类
        class subDataset(Dataset.Dataset):
            # 初始化，定义数据内容和标签
            def __init__(self, teatData, testLabel):
                self.testData = testData
                self.testLabel = testLabel

            # 返回数据集大小
            def __len__(self):
                return len(self.testData)

            # 得到数据内容和标签
            def __getitem__(self, index):
                testdata = torch.LongTensor(self.testData[index])
                testlabel = torch.LongTensor(self.testLabel[index])
                return testdata, testlabel


        if __name__ == '__main__':
            testdataset = subDataset(testData, testLabel)
            # print(dataset)
            print('dataset大小为：', testdataset.__len__())
            # print(dataset.__getitem__(0))
            # print(dataset[0])

            # 创建DataLoader迭代器

            testdataloader = DataLoader.DataLoader(testdataset, batch_size=500, shuffle=True, num_workers=4)

        for i, (testdata, testlabel) in enumerate(testdataloader):
            data = testdata.float().cuda()
            label = testlabel.cuda()
            test_output = classi(data.view(-1, layer,1))  # (samples, time_step, input_size)

            label = label.flatten()
            out = torch.squeeze(test_output)
            loss = loss_func2(out, label)
            if i % 50 == 0:
                print('Epoch:', epoch, '|test loss %.4f' % loss.cpu().data.numpy())

            test_output = test_output.cpu()

            pred_y = torch.max(torch.squeeze(test_output), 1)[1].data.numpy()

            label = label.cpu()
            target_y = label.data.numpy()
            # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
            accuracy = accuracy_score(target_y, pred_y)  # 预测中有多少和真实值一样
            recall = recall_score(target_y, pred_y)
            f1score = f1_score(target_y, pred_y)
            precision = precision_score(target_y, pred_y)

            score = torch.squeeze(test_output).data.numpy()[:, 1]
            # score = np.amax(score,axis=1)

            fpr, tpr, threshold = roc_curve(target_y, score)  ###计算真正率和假正率
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)  ###计算auc的值
            aucs.append(roc_auc)

            # pre, re, th = precision_recall_curve(label, score)
            average_precision = average_precision_score(label, score)
            # precs.append(interp(mean_recall, re, pre))
            # precs[-1][0] = 0.0
            # roc_auc2 = auc(re, pre)  ###计算auc的值
            # aucs2.append(roc_auc2)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, )
            # plt.plot(re, pre, lw=1, alpha=0.3, )
            print('Accuracy:%.4f' % accuracy, 'Recall:%.4f' % recall, 'Precision:%.4f' % precision, 'F1:%.4f' % f1score)
            print('ROCAUC:%.4f' % roc_auc, 'PRAUC:%.4f' % average_precision)

            Accuracy.append(accuracy)
            Recall.append(recall)
            Precision.append(precision)
            F1score.append(f1score)
            Average_precision.append(average_precision)
            k += 1


mean_tpr=np.mean(tprs,axis=0)
mean_tpr[-1]=1.0
mean_auc=auc(mean_fpr,mean_tpr)#计算平均AUC值
std_auc=np.std(tprs,axis=0)
plt.plot(mean_fpr,mean_tpr,color='b',label=r' Model(Full) (mean_AUROC=%0.3f)'%mean_auc,lw=2,alpha=.8)
std_tpr=np.std(tprs,axis=0)
tprs_upper=np.minimum(mean_tpr+std_tpr,1)
tprs_lower=np.maximum(mean_tpr-std_tpr,0)
plt.fill_between(mean_tpr,tprs_lower,tprs_upper,color='gray',alpha=.2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

Accuracy = np.mean(Accuracy)
Recall = np.mean(Recall)
Precision = np.mean(Precision)
F1score = np.mean(F1score)
Average_precision = np.mean(Average_precision)
print('Accuracy:%.4f' % Accuracy, 'Recall:%.4f' % Recall, 'Precision:%.4f' % Precision, 'F1:%.4f' % F1score)
print('ROCAUC:%.4f' % mean_auc, 'PRAUC:%.4f' % Average_precision)
