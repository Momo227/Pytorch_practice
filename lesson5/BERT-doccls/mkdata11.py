#!/usr/bin/python
# -*- coding: sjis -*-

from transformers import BertJapaneseTokenizer
import pickle
import re

tknz = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')

xdata, ydata = [],[]
with open('train.tsv','r',encoding='utf-8') as f:
    for line in f:
        line = line.rstrip()
        # re.match：文字列の先頭がマッチするかチェック、抽出
        # ^:文字列の先頭, \d:任意の数字, +:１回以上の繰り返し, \t:空白文字ではない任意の文字, .:任意の一文字, ?:０回または１回, $:文字列の末尾
        result = re.match('^(\d+)\t(.+?)$', line)
        # group：マッチした文字列を取得, (1)：1番目にマッチ
        ydata.append(int(result.group(1)))
        sen = result.group(2)
        tid = tknz.encode(sen)
        if (len(tid) > 512):  # 最大長は 512
            tid = tid[:512]
        xdata.append(tid)

with open('xtrain.pkl','bw') as fw:
    pickle.dump(xdata,fw)

with open('ytrain.pkl','bw') as fw:
    pickle.dump(ydata,fw)
