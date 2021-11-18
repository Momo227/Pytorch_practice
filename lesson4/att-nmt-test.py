#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys

argvs = sys.argv
argc = len(argvs)

# GPU�ϊ�
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data setting
# en-id�ǂݍ���
id, eid2w, ew2id = 1, {}, {}
with open('data/train.en.vocab.4k','r',encoding='utf-8') as f:
    for w in f:
        w = w.strip()
        eid2w[id] = w
        ew2id[w] = id
        id += 1
ev = id

# en-test�f�[�^�̓ǂݍ���
edata = []
with open('data/test.en','r',encoding='utf-8') as f:
    for sen in f:
        wl = [ew2id['<s>']]
        for w in sen.strip().split():
            if w in ew2id:
                wl.append(ew2id[w])
            else:
                wl.append(ew2id['<unk>'])
        wl.append(ew2id['</s>'])
        edata.append(wl)

# ja-id�ǂݍ���
id, jid2w, jw2id = 1, {}, {}
with open('data/train.ja.vocab.4k','r',encoding='utf-8') as f:
    id = 1
    for w in f:
        w = w.strip()
        jid2w[id] = w
        jw2id[w] = id
        id += 1
jv = id

# ja-test�f�[�^�̓ǂݍ���
jdata = []
with open('data/test.ja','r',encoding='utf-8') as f:
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

class MyAttNMT(nn.Module):
    def __init__(self, jv, ev, k):
        super(MyAttNMT, self).__init__()
        # �P�ꖄ�ߍ���
        self.jemb = nn.Embedding(jv, k)
        self.eemb = nn.Embedding(ev, k)
        # lstm2�w
        self.lstm1 = nn.LSTM(k, k, num_layers=2,
                             batch_first=True)
        self.lstm2 = nn.LSTM(k, k, num_layers=2,
                             batch_first=True)
        self.Wc = nn.Linear(2*k, k)
        self.W = nn.Linear(k, ev)
    def forward(self, jline, eline):
        x = self.jemb(jline)
        ox, (hnx, cnx) = self.lstm1(x)
        y = self.eemb(eline)
        oy, (hny, cny) = self.lstm2(y,(hnx, cnx))
        ox1 = ox.permute(0,2,1)
        sim = torch.bmm(oy,ox1)
        bs, yws, xws = sim.shape
        sim2 = sim.reshape(bs*yws,xws)
        alpha = F.softmax(sim2,dim=1).reshape(bs, yws, xws)
        ct = torch.bmm(alpha,ox)
        oy1 = torch.cat([ct,oy],dim=2)
        oy2 = self.Wc(oy1)
        return self.W(oy2)

# model generate, optimizer and criterion setting

demb = 200
net = MyAttNMT(jv, ev, demb).to(device)

net.load_state_dict(torch.load(argvs[1]))

# Translate

esid = ew2id['<s>']
eeid = ew2id['</s>']
net.eval()
with torch.no_grad():
    for i in range(len(jdata)):
        # attention�̏������܂�
        jinput = torch.LongTensor([jdata[i][1:]]).to(device)
        x = net.jemb(jinput)
        # lstm�̏���
        ox, (hnx, cnx) = net.lstm1(x)  # �o�b�`�T�C�Y, �󕶂̒P�ꐔ, ���ԕ\���̎�����
        wid = esid
        sl = 0
        while True:
            wids = torch.LongTensor([[wid]]).to(device)
            y = net.eemb(wids)

            oy, (hnx, cnx) = net.lstm2(y,(hnx, cnx))  # �o�b�`�T�C�Y, �󕶂̒P�ꐔ, ���ԕ\���̎�����
            ox1 = ox.permute(0,2,1)
            # ����
            sim = torch.bmm(oy,ox1)
            bs, yws, xws = sim.shape
            sim2 = sim.reshape(bs*yws,xws)

            alpha = F.softmax(sim2,dim=1).reshape(bs, yws, xws)
            ct = torch.bmm(alpha,ox)
            # cat:�A������Tensor��Ԃ�
            # torch.cat(tensors, dim=0, *, out=None) �� Tensor
            oy1 = torch.cat([ct,oy],dim=2)
            oy2 = net.Wc(oy1)
            oy3 = net.W(oy2)
            wid = torch.argmax(oy3[0]).item()
            if (wid == eeid):
                break
            print(eid2w[wid]," ",end='')
            sl += 1
            if (sl == 30):
                break
        print()
