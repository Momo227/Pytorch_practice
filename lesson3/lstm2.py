#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('dic.pkl', 'br') as f:
    dic = pickle.load(f)

labels = {'名詞': 0, '助詞': 1, '形容詞': 2,
     '助動詞': 3, '補助記号': 4, '動詞': 5, '代名詞': 6,
     '接尾辞': 7, '副詞': 8, '形状詞': 9, '記号': 10,
     '連体詞': 11, '接頭辞': 12, '接続詞': 13,
     '感動詞': 14, '空白': 15}

# data の設定

class MyDataset(Dataset):
    def __init__(self, xdata, ydata):
        self.data = xdata
        self.label = ydata
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        return (x,y)

def my_collate_fn(batch):
    images, targets= list(zip(*batch))
    xs = list(images)
    ys = list(targets)
    return xs, ys

with open('xtrain.pkl', 'br') as fr:
    xdata = pickle.load(fr)

with open('ytrain.pkl', 'br') as fr:
    ydata = pickle.load(fr)

batch_size = 200
dataset = MyDataset(xdata,ydata)
# DataLoader : datasetsからバッチごとに取り出す
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)

# model の定義

class MyLSTM(nn.Module):
    def __init__(self, vocsize, posn, hdim):
        super(MyLSTM, self).__init__()
        self.embd = nn.Embedding(vocsize, hdim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=hdim, hidden_size=hdim, batch_first=True, num_layers=2)
        self.ln   = nn.Linear(hdim, posn)
    def forward(self, x):
        x = self.embd(x)
        lo, (hn, cn) = self.lstm(x)
        out = self.ln(lo)
        return out

# model generate, optimizer, criterion の設定

net = MyLSTM(len(dic)+1, len(labels), 100).to(device)
optimizer = optim.SGD(net.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=-1)

# 訓練
net.train()
for ep in range(10):
    loss10B, i = 0.0, 0
    for xs, ys in dataloader:
        xs1, ys1 = [], []
        for k in range(len(xs)):
            tid = xs[k]
            xs1.append(torch.LongTensor(tid))
            tid = ys[k]
            ys1.append(torch.LongTensor(tid))
        xs1 = pad_sequence(xs1, batch_first=True).to(device)
        ys1 = pad_sequence(ys1, batch_first=True, padding_value=-1.0)
        output = net(xs1)
        ys1 = ys1.type(torch.LongTensor).to(device)
        loss = criterion(output[0],ys1[0])
        for h in range(1,len(ys1)):
            loss += criterion(output[h],ys1[h])
        if (i % 10 == 0):
            print(ep, i, loss10B)
            loss10B = 0.0
        else:
            loss10B += loss.item()
        i += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    outfile = "lstm2-" + str(ep) + ".model"
    # state_dict : 各層をそのパラメータ tensor にマップする Python 辞書
    torch.save(net.state_dict(),outfile)
