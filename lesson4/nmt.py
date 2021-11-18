#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data setting

# �f�[�^��P��ɕ������Aid�ɒP��A�P���id���֘A�Â���
# {}:����, []:list
id, eid2w, ew2id = 1, {}, {}
with open('data/train.en.vocab.4k','r',encoding='utf-8') as f:
    for w in f:
        w = w.strip()
        eid2w[id] = w
        ew2id[w] = id
        id += 1
ev = id

edata = []
with open('data/train.en','r',encoding='utf-8') as f:
    for sen in f:
        # ������"<s>"��id
        wl = [ew2id["<s>"]]
        for w in sen.strip().split():
            #  ���ԂɒP���id
            if w in ew2id:
                wl.append(ew2id[w])
            else:
                wl.append(ew2id['<unk>'])
        # ������"</s>"��id
        wl.append(ew2id['</s>'])
        edata.append(wl)

id, jid2w, jw2id = 1, {}, {}
with open('data/train.ja.vocab.4k','r',encoding='utf-8') as f:
    id = 1
    for w in f:
        w = w.strip()
        jid2w[id] = w
        jw2id[w] = id
        id += 1
jv = id

jdata = []
with open('data/train.ja','r',encoding='utf-8') as f:
    for sen in f:
        wl = [jw2id['<s>']]
        for w in sen.strip().split():
            if w in jw2id:
                wl.append(jw2id[w])
            else:
                wl.append(jw2id['<unk>'])
        wl.append(jw2id['</s>'])
        jdata.append(wl)

# Define model

class MyNMT(nn.Module):
    def __init__(self, jv, ev, k):
        super(MyNMT, self).__init__()
        self.jemb = nn.Embedding(jv, k)
        self.eemb = nn.Embedding(ev, k)
        # ����k����, �B����k����, 2�w��RNN  (�o����LSTM:bidirectional=True)
        self.lstm1 = nn.LSTM(k, k, num_layers=2,
                             batch_first=True)
        self.lstm2 = nn.LSTM(k, k, num_layers=2,
                             batch_first=True)
        self.W = nn.Linear(k, ev)
    # Softmax�͑����̌v�Z�ōs��
    def forward(self, jline, eline):
        x = self.jemb(jline)
        ox, (hnx, cnx) = self.lstm1(x)
        y = self.eemb(eline)
        oy, (hny, cny) = self.lstm2(y,(hnx, cnx))
        out = self.W(oy)
        return out

# model generate, optimizer and criterion setting

demb = 200
net = MyNMT(jv, ev, demb).to(device)
optimizer = optim.SGD(net.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()

# Learn

net.train()
for epoch in range(20):
    loss1K = 0.0
    for i in range(len(jdata)):
        # ������"<s>"�ƕ�����"</s>"��input���Ȃ�
        jinput = torch.LongTensor([jdata[i][1:]]).to(device)
        einput = torch.LongTensor([edata[i][:-1]]).to(device)
        out = net(jinput, einput)
        # [[1, 2, 3]]�ɂȂ��Ă���
        gans = torch.LongTensor([edata[i][1:]]).to(device)
        # out[0],gans[0] -> [0]�ɂ��邱�Ƃ�[]��1�O��
        loss = criterion(out[0],gans[0])
         # loss.item()�ɂ��邱�ƂŁA���l�̂ݓ�����i�Ȃ����tensor(8.3177, grad_fn=<NllLossBackward>)�j
        loss1K += loss.item()
        if (i % 100 == 0):
            print(epoch, i, loss1K)
            loss1K = 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    outfile = "outputs/models/nmt-" + str(epoch) + ".model"
    torch.save(net.state_dict(),outfile)
