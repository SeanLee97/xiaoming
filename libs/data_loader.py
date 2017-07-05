# !/usr/bin/env python
# -*- coding: utf-8 -*-

'''
加载语料
返回 Variable
'''
import os, math, random, re, time, sys
import torch
from torch.autograd import Variable
from io import open
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from libs.get_files import getFIles
from libs.func import *
from config.config  import *
import torchwordemb

q_file = CORPUS_DIR + '/q.txt'
a_file = CORPUS_DIR + '/a.txt'

class dataLoader(object):
	def __init__(self):
		# 加载词库
		if os.path.exists(DICT_PATH):
			print('加载词典中...')
			jieba.load_userdict(DICT_PATH)
		self.load_corpus()

	# 内部类
	class Lang(object):
		def __init__(self):
			self.index2word = {SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"}
			self.word2index = {}
			self.n_words = 2

		def set_nwords(self, n_words):
			self.n_words += n_words

		def set_word2index(self, word2index):
			self.word2index = word2index

		def set_index2word(self, index2word):
			self.index2word = dict(self.index2word)
			self.index2word.update(index2word)

	def filter_line(self, line):
		word_list = wordseg(line)
		return len(word_list) < MAX_LENGTH
	
	def load_corpus(self):
		print("正在加载语料库...")
		
		# 判断文件是否存在		
		if not os.path.exists(q_file):
			print("请将question库文件名命名为q.txt，quesion库目录应为%s" %  q_file)
			sys.exit()
		if not os.path.exists(a_file):
			print("请将answer库文件名命名为q.txt，anwser库目录应为%s" % a_file)
			sys.exit()
		
		'''word2vec'''

		self.word2vec()

		vocab, _ = torchwordemb.load_word2vec_bin(WORD2VEC_PATH)

		# 加载语料并处理
		self.en_lang = self.Lang() # encoder_lang
		self.de_lang = self.Lang() # decoder_lang
		
		q_lines = []
		a_lines = []
		with open(q_file, 'r') as f:
			for line in f.readlines():
				q_lines.append(line.strip('\n'))
		with open(a_file, 'r') as f:
			for line in f.readlines():
				a_lines.append(line.strip('\n'))
		self.pairs = []
		for i in range(len(q_lines)):
			self.pairs.append({0: q_lines[i], 1: a_lines[i]})

		en_nwords = 0
		de_nwords = 0
		en_index2word = {}
		en_word2index = {}
		de_index2word = {}
		de_word2index = {}
		self.word_dict = {}

		for k,v in vocab.items():
			self.word_dict[k] = int(v)+2 #将词后移动，使头两位为SOS，EOS
		word_index = 0
		for en_line in q_lines:
			word_list = wordseg(en_line)
			for word in word_list:
				if word in self.word_dict:
					word_index = self.word_dict[word]
					en_word2index[word] = word_index
					en_index2word[word_index] = word
					en_nwords+=1

		self.en_lang.set_word2index(en_word2index)
		self.en_lang.set_index2word(en_index2word)
		self.en_lang.set_nwords(en_nwords)

		for de_line in a_lines:
			word_list = wordseg(de_line)
			for word in word_list:
				if word in self.word_dict:
					word_index = self.word_dict[word]
					de_word2index[word] = word_index
					de_index2word[word_index] = word
					de_nwords+=1

		self.de_lang.set_word2index(de_word2index)
		self.de_lang.set_index2word(de_index2word)
		self.de_lang.set_nwords(de_nwords)

		print("语料库统计：")
		print("Q: %d 词" % self.en_lang.n_words)
		print("A: %d 词" % self.de_lang.n_words)

	def word2vec(self):
		print("word2vec ing...")

		sentences = getFIles(CORPUS_DIR)
		model = Word2Vec(sentences, min_count = 0)
		model.wv.save_word2vec_format(WORD2VEC_PATH, binary=True)
		print("end")

	def indexes_from_sentence(self, lang, sentence):
		word_list = wordseg(sentence)
		index_list = [];
		for word in word_list:
			if word in self.word_dict:
				index_list.append(self.word_dict[word])
		if len(index_list) == 0:
				index_list = [0]
		return list(index_list)
		#return [lang.word2index[word] for word in word_list]
		
		
	def variable_from_sentence(self, lang, sentence):
		indexes = self.indexes_from_sentence(lang, sentence)
		indexes.append(EOS_TOKEN)
		result = Variable(torch.LongTensor(indexes).view(-1,1))
		if USE_CUDA:
			return result.cuda()
		else:
			return result

	def get_pair_variable(self):
		pair = random.choice(self.pairs)
		en_variable = self.variable_from_sentence(self.en_lang, pair[0])
		de_variable = self.variable_from_sentence(self.de_lang, pair[1])
		return en_variable, de_variable

if __name__  == '__main__':
	pass
