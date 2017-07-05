# !/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import os
import glob
import numpy as np
import time
import math
from rnn_models import *
from config.config  import *


def load_previous_model(encoder, decoder):
    f_list = glob.glob(os.path.join(CHECKPOINT_DIR, MODEL_PREFIX) + '-*.ckpt')
    start_epoch = 1
    if len(f_list) >= 1:
        epoch_list = [int(i.split('-')[-1].split('.')[0]) for i in f_list]
        last_checkpoint = f_list[np.argmax(epoch_list)]
        if os.path.exists(last_checkpoint):
            if DEBUG:
                print('load from {}'.format(last_checkpoint))
            model_state_dict = torch.load(last_checkpoint, map_location=lambda storage, loc: storage)
            encoder.load_state_dict(model_state_dict['encoder'])
            decoder.load_state_dict(model_state_dict['decoder'])
            start_epoch = np.max(epoch_list)
    return encoder, decoder, start_epoch

def load_last_model():
    f_list = glob.glob(os.path.join(CHECKPOINT_DIR, MODEL_PREFIX) + '-*.ckpt')
    start_epoch = 1
    if len(f_list) >= 1:
        epoch_list = [int(i.split('-')[-1].split('.')[0]) for i in f_list]
        last_checkpoint = f_list[np.argmax(epoch_list)]
        if os.path.exists(last_checkpoint):
            if DEBUG:
                print('load from {}'.format(last_checkpoint))
            model_state_dict = torch.load(last_checkpoint, map_location=lambda storage, loc: storage)
            encoder = EncoderRNN(model_state_dict['input_size'], HIDDEN_SIZE, N_LAYERS)
            decoder = AttnDecoderRNN(ATTN_MODEL, HIDDEN_SIZE, model_state_dict['output_size'],
                                   N_LAYERS, dropout_p=DROPOUT_P)
            encoder.load_state_dict(model_state_dict['encoder'])
            decoder.load_state_dict(model_state_dict['decoder'])
            start_epoch = np.max(epoch_list)
    return encoder, decoder, start_epoch

def save_model(encoder, decoder, data_loader, epoch, max_keep=5):
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    f_list = glob.glob(os.path.join(CHECKPOINT_DIR, MODEL_PREFIX) + '-*.ckpt')
    if len(f_list) >= max_keep + 2:
        # this step using for delete the more than 5 and litter one
        epoch_list = [int(i.split('-')[-1].split('.')[0]) for i in f_list]
        to_delete = [f_list[i] for i in np.argsort(epoch_list)[-max_keep:]]
        for f in to_delete:
            os.remove(f)
    name = MODEL_PREFIX + '-{}.ckpt'.format(epoch)
    file_path = os.path.join(CHECKPOINT_DIR, name)
    model_dict = {
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'hidden_size': HIDDEN_SIZE,
        'max_length': MAX_LENGTH,
        'dropout_p': DROPOUT_P,
        'n_layers': N_LAYERS,
        'input_size': data_loader.en_lang.n_words,
        'output_size': data_loader.de_lang.n_words,
    }
    torch.save(model_dict, file_path)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    if percent == 0:
    	percent = 0.000001
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))
