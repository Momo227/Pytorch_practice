#!/usr/bin/python
# -*- coding: sjis -*-
# 品詞タグ付けして精度評価

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

# data の設定
with open('xtest.pkl', 'br') as f:
    xtest = pickle.load(f)

with open('ytest.pkl', 'br') as f:
    ytest = pickle.load(f)

# model の定義

class MyLSTM(nn.Module):
    def __init__(self, vocsize, posn, hdim): # posn:position
        super(MyLSTM, self).__init__()
        self.embd = nn.Embedding(vocsize, hdim)
        # batch_first=True : (seq_len, batch, input_size)と指定されている入力テンソルの型を(batch, seq_len, input_size)にする
        self.lstm = nn.LSTM(input_size=hdim, hidden_size=hdim, batch_first=True)
        self.ln   = nn.Linear(hdim, posn)
    def forward(self, x):
        x = self.embd(x)
        lo, (hn, cn) = self.lstm(x)
        out = self.ln(lo)
        return out

# model generate, optimizer, criterion の設定

net = MyLSTM(len(dic)+1, len(lab), 100).to(device)
# CPUマシンでのGPUモデルの読み出し
net.load_state_dict(torch.load(argvs[1]))

# 評価
real_data_num = 0  # データの個数
# 学習/推論のモードが切り替わりDropoutなどの挙動に影響する。
net.eval()
# 1回きり
with torch.no_grad():
    ok = 0  # ok は正解数
    for i in range(len(xtest)):
        real_data_num += len(xtest[i])
        x = [ xtest[i] ]
        x = torch.LongTensor(x).to(device)
        output = net(x)
        ans = torch.argmax(output[0],dim=1)
        y = torch.LongTensor(ytest[i]).to(device)
        ok += torch.sum(ans == y).item()
    print(ok, real_data_num, ok/real_data_num)
