#!/usr/bin/env python
#-*-coding: utf-8 -*-

import os
import math
import random
import re
import time
import unicodedata
from io import open
import torch
from torch.autograd import Variable
import torch.optim as optim
from rnn_models import *
from libs.model_utils import *
from libs.data_loader import dataLoader
from config.config  import *


def evaluate(data_loader, encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = data_loader.variable_from_sentence(data_loader.en_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.init_hidden()

    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    decoder_input = Variable(torch.LongTensor([[SOS_TOKEN]])) # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_input = Variable(torch.LongTensor([[SOS_TOKEN]])) # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    decoder_hidden = encoder_hidden
    
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_TOKEN:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(data_loader.de_lang.index2word[ni])
            
        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()
    
    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]


def evaluate_randomly(data_loader, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(data_loader.pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(data_loader, encoder, decoder, 'he s waiting for you at home')
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

'''
def read_voca(input_file):
    tmp_vocab = []
    with open(input_file, "r") as f:
        tmp_vocab.extend(f.readlines())
        tmp_vocab = [line.strip() for line in tmp_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
        return vocab, tmp_vocab
'''

def main():  
    #data_loader = dataLoader() 
    data_loader = dataLoader()
    encoder1 = EncoderRNN(data_loader.en_lang.n_words, HIDDEN_SIZE, N_LAYERS)
    attn_decoder1 = AttnDecoderRNN(ATTN_MODEL, HIDDEN_SIZE, data_loader.de_lang.n_words,
                                   N_LAYERS, dropout_p=DROPOUT_P)
    encoder, decoder, start_epoch = load_previous_model(encoder1, attn_decoder1)
    while True:
        input_string = input('me > ')
        # 退出
        if input_string == 'quit':
            exit()
        output_words, attentions = evaluate(data_loader, encoder, decoder, input_string)
        output_sentence = ''
        for word in output_words:
            output_sentence += word;
        print(output_sentence)
    '''
    data_loader, encoder, decoder, start_epoch = load_last_model(CHECKPOINT_DIR, MODEL_PREFIX)
    while True:
        input_string = input('me > ')
        # 退出
        if input_string == 'quit':
            exit()
        output_words, attentions = evaluate(data_loader, encoder, decoder, input_string)
        output_sentence = ''
        for word in output_words:
            output_sentence += word;
        print(output_sentence)
    '''
if __name__ == '__main__':
    main()
