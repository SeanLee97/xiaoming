# !/usr/bin/env python
# -*- coding: utf-8 -*-
'''
配置文件
'''

import torch
import os


DEBUG 		 		= False
MODEL_PREFIX	 	= 'seq2seq_attn_'
CORPUS_DIR	 		= './corpus/qa1'
RUNTIME_DIR 		= './runtime'
COMMON_DIR			= './common'
CHECKPOINT_DIR	 	= RUNTIME_DIR + '/model'	#save model to dir


DICT_PATH			= COMMON_DIR  + '/word_dict.txt'
WORD2VEC_PATH 		= RUNTIME_DIR + '/word2vec/word2vec.model.bin'
#WORD2VEC_PATH 		= '/root/Project/python3/xiaoming/runtime/word2vec/word2vec.model.bin'
SOS_TOKEN 			= 0
EOS_TOKEN 			= 1

'''
net parameters
'''
USE_CUDA 			= torch.cuda.is_available()
LEARNING_RATE 		= 0.01
LEARNING_FORCE_RATE = 0.5
CLIP				= 5.0
N_EPOCHS			= 10000
N_LAYERS			= 2
HIDDEN_SIZE			= 256
MAX_LENGTH	 		= 60
DROPOUT_P 			= 0.1
ATTN_MODEL			= 'general'