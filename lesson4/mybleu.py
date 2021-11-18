#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from nltk.translate.bleu_score import corpus_bleu


argvs = sys.argv

gold = []
with open('data/test.en', 'r') as f:
    for sen in f:
        w = sen.strip().split()
        gold.append([w])

myans = []
with open(argvs[1], 'r') as f:
    for sen in f:
        w = sen.strip().split()
        myans.append(w)

# データは両方、単語の集合の文（list）
score = corpus_bleu(gold, myans)
print(100 * score)
