# !/usr/bin/env python
# -*- coding: utf-8 -*-

'''
以迭代器的方式读取文件节省RAM空间
'''
from io import open
import os
import jieba
from config.config  import *
from libs.func import *

class getFIles(object):
	def __init__(self, dirname):
		self.dirname = dirname
		if os.path.exists(DICT_PATH):
			jieba.load_userdict(DICT_PATH)

	def __iter__(self):
		for filename in os.listdir(self.dirname):
			for line in open(os.path.join(self.dirname, filename)):
				yield wordseg(line)

if __name__ == '__main__':
	sentences = getFIles('/path/to/file')