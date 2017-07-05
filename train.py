# !/usr/bin/env python
# -*- coding: utf-8 -*-
'''

'''
import math
import random
import os,re
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

def train_model(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer,
                criterion, max_length=MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    
    decoder_input = Variable(torch.LongTensor([[SOS_TOKEN]]))
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    use_teacher_forcing = True if random.random() < LEARNING_RATE else False

    if use_teacher_forcing:
        try:
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output[0], target_variable[di])
                decoder_input = target_variable[di]
        except KeyboardInterrupt:
            return

    else:
        # 
        try:
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output[0], target_variable[di])
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

                loss += criterion(decoder_output[0], target_variable[di])
                if ni == EOS_TOKEN:
                    break
        except KeyboardInterrupt:
            return

    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), CLIP)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), CLIP)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def train(data_loader, encoder, decoder, print_every=100, save_every=1000, evaluate_every=100,
          learning_rate=0.01):
    start = time.time()
    print_loss_total = 0 

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    encoder, decoder, start_epoch = load_previous_model(encoder, decoder)

    for epoch in range(start_epoch, N_EPOCHS + 1):
        input_variable, target_variable = data_loader.get_pair_variable()
        #print(input_variable)
        try:
            loss = train_model(input_variable, target_variable, encoder,
                               decoder, encoder_optimizer, decoder_optimizer, criterion)
        except KeyboardInterrupt:
            pass
        print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, epoch / N_EPOCHS),
                                         epoch, epoch / N_EPOCHS * 100, print_loss_avg))

        if epoch % save_every == 0:
            save_model(encoder, decoder, data_loader, epoch)

        if epoch % evaluate_every == 0:
            evaluate_randomly(data_loader, encoder, decoder, n=1)


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
    nl = [i + 1 for i in range(1, n+1)]
    for i in range(n):
        # 随机抽取
        pair = random.choice(data_loader.pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(data_loader, encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def main():
    data_loader = dataLoader()
    encoder = EncoderRNN(data_loader.en_lang.n_words, HIDDEN_SIZE, N_LAYERS)
    decoder = AttnDecoderRNN(ATTN_MODEL, HIDDEN_SIZE, data_loader.de_lang.n_words,
                                   N_LAYERS, dropout_p=DROPOUT_P)

    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    print('start training...')
    train(data_loader, encoder, decoder)

if __name__ == '__main__':
    main()
