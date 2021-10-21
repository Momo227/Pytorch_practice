#!/usr/bin/python
# -*- coding: sjis -*-
# �i���^�O�t�����Đ��x�]��

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import pickle

argvs = sys.argv
argc = len(argvs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('dic.pkl', 'br') as f:
    dic = pickle.load(f)

with open('label.pkl', 'br') as f:
    lab = pickle.load(f)

# data �̐ݒ�
with open('xtest.pkl', 'br') as f:
    xtest = pickle.load(f)

with open('ytest.pkl', 'br') as f:
    ytest = pickle.load(f)

# model �̒�`

class MyLSTM(nn.Module):
    def __init__(self, vocsize, posn, hdim): # posn:position
        super(MyLSTM, self).__init__()
        self.embd = nn.Embedding(vocsize, hdim)
        # batch_first=True : (seq_len, batch, input_size)�Ǝw�肳��Ă�����̓e���\���̌^��(batch, seq_len, input_size)�ɂ���
        self.lstm = nn.LSTM(input_size=hdim, hidden_size=hdim, batch_first=True)
        self.ln   = nn.Linear(hdim, posn)
    def forward(self, x):
        x = self.embd(x)
        lo, (hn, cn) = self.lstm(x)
        out = self.ln(lo)
        return out

# model generate, optimizer, criterion �̐ݒ�

net = MyLSTM(len(dic)+1, len(lab), 100).to(device)
# CPU�}�V���ł�GPU���f���̓ǂݏo��
net.load_state_dict(torch.load(argvs[1]))

# �]��
real_data_num = 0  # �f�[�^�̌�
# �w�K/���_�̃��[�h���؂�ւ��Dropout�Ȃǂ̋����ɉe������B
net.eval()
# 1�񂫂�
with torch.no_grad():
    ok = 0  # ok �͐���
    for i in range(len(xtest)):
        real_data_num += len(xtest[i])
        x = [ xtest[i] ]
        x = torch.LongTensor(x).to(device)
        output = net(x)
        ans = torch.argmax(output[0],dim=1)
        y = torch.LongTensor(ytest[i]).to(device)
        ok += torch.sum(ans == y).item()
    print(ok, real_data_num, ok/real_data_num)
