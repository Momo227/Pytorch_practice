#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertModel, BertConfig

import numpy as np
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data setting

with open('xtrain.pkl','br') as fr:
    xtrain = pickle.load(fr)

with open('ytrain.pkl','br') as fr:
    ytrain = pickle.load(fr)

# Define model

bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')

class DocCls(nn.Module):
    def __init__(self,bert):
        super(DocCls, self).__init__()
        self.bert = bert
        self.cls=nn.Linear(768,9)
    def forward(self,x):
        bout = self.bert(x)
        # バッチサイズの取り出し
        bs = len(bout[0])
        # バッチ内i番目の文に対する[CLS]の埋め込み表現
        # BERTは双方向→[CLS]：多層エンコード手順を通じてすべてのトークンの代表的な情報を含む。
        h0 = [ bout[0][i][0] for i in range(bs)]
        # stack:連結
        h0 = torch.stack(h0,dim=0)
        return self.cls(h0)

# model generate, optimizer and criterion setting

net = DocCls(bert).to(device)
optimizer = optim.SGD(net.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()

# Learn

net.train()
# 30エポックまで学習
for ep in range(30):
    lossK = 0.0
    for i in range(len(xtrain)):
        x = torch.LongTensor(xtrain[i]).unsqueeze(0).to(device)
        y = torch.LongTensor([ ytrain[i] ]).to(device)
        out = net(x)
        loss = criterion(out,y)
        lossK += loss.item()
        # 50データごとに損失値の合計を評
        if (i % 50 == 0):
            print(ep, i, lossK)
            lossK = 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    outfile = "../outputs/doccls-" + str(ep) + ".model"
    torch.save(net.state_dict(),outfile)
