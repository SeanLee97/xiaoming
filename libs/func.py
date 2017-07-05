# !/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import jieba

# 去除标点符号 
def rm_sign(strs):
	return re.sub("[\s+\.\!\/<>“”,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", strs.strip())

def wordseg(sentence, rmsign=True):
	sentence = sentence.strip()
	if rmsign:
		sentence = rm_sign(sentence)
	word_list = jieba.lcut(sentence.strip())
	return word_list