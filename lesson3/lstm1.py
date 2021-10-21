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

labels = {'����': 0, '����': 1, '�`�e��': 2,
     '������': 3, '�⏕�L��': 4, '����': 5, '�㖼��': 6,
     '�ڔ���': 7, '����': 8, '�`��': 9, '�L��': 10,
     '�A�̎�': 11, '�ړ���': 12, '�ڑ���': 13,
     '������': 14, '��': 15}

# data �̐ݒ�

class MyDataset(Dataset):
    def __init__(self, xdata, ydata):
        self.data = xdata
        self.label = ydata
    # �g�ݍ��݊֐�len()�̗��p
    def __len__(self):
        return len(self.label)
    # �����^�g�p���ŁA����L�[�����o����ƃG���[�����o�͂���v���O����
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        return x, y

def my_collate_fn(batch):
    xdata, ydata = list(zip(*batch))
    xs = list(xdata)
    ys = list(ydata)
    return xs, ys
    
# def my_collate_fn(batch):
#     images, targets= list(zip(*batch))
#     xs = list(images)
#     ys = list(targets)
#     return xs, ys

with open('xtrain.pkl', 'br') as fr:
    xdata = pickle.load(fr)

with open('ytrain.pkl', 'br') as fr:
    ydata = pickle.load(fr)

batch_size = 200
dataset = MyDataset(xdata,ydata)
# collate_fn:dataset�Œ�`���ꂽ__getitem__���o�b�`�̌`�ɂȂ�Ƃ��A���ꂼ��̗v�f�����X�g�Ōł߂���B
# ����𑀍삵�A�ŏI�I�ɂ�torch.Tensor�ɂ���֐�
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)

# model �̒�`

class MyLSTM(nn.Module):
    def __init__(self, vocsize, posn, hdim):
        super(MyLSTM, self).__init__()
        self.embd = nn.Embedding(vocsize, hdim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=hdim, hidden_size=hdim, batch_first=True)
        self.ln   = nn.Linear(hdim, posn)
    def forward(self, x):
        x = self.embd(x)
        lo, (hn, cn) = self.lstm(x)
        out = self.ln(lo)
        return out

# class MyLSTM(nn.Module):
#     def __init__(self, vocsize, posn, hdim):
#         super(MyLSTM, self).__init__()
#         self.embd = nn.Embedding(vocsize, hdim, padding_idx=0)
#         self.lstm = nn.LSTM(input_size=hdim, hidden_size=hdim)
#         self.ln   = nn.Linear(hdim, posn)
#     def forward(self, x):
#         x = self.embd(x)
#         lo, (hn, cn) = self.lstm(x)
#         out = self.ln(lo)
#         return out

# model generate, optimizer, criterion �̐ݒ�

net = MyLSTM(len(dic)+1, len(labels), 100).to(device)
optimizer = optim.SGD(net.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=-1)

# �P��

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
        xs1 = pad_sequence(xs1, batch_first=True).to(device) # padding_value�̎w�肪�Ȃ� = 0
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
        optimizer.zero_grad() # �p�����[�^�̌��z��������
        loss.backward() # �e�p�����[�^�̌��z���Z�o
        optimizer.step() # ���z�̏���p�����p�����[�^�̍X�V
    outfile = "lstm1-" + str(ep) + ".model"
    torch.save(net.state_dict(),outfile)
