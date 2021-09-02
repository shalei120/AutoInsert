# Copyright 2020 . All Rights Reserved.
# Author : Lei Sha
import functools
print = functools.partial(print, flush=True)
from TranslationModel import TranslationModel

from textdata import  TextData_MT
import time, sys,datetime
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import  tqdm
import os
import math,random
import nltk
from nltk.corpus import stopwords
import argparse
# import GPUtil
# import turibolt as bolt
import pickle, json
from Hyperparameters_MT import args
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# import matplotlib.pyplot as plt
import numpy as np
import copy
from transformers import Trainer, TrainingArguments
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
# from transformers.modeling_bart import shift_tokens_right
from transformers import LongformerTokenizer, EncoderDecoderModel
from transformers import XLNetTokenizer, XLNetModel
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
parser.add_argument('--batch', '-b')
parser.add_argument('--modelarch', '-m')
parser.add_argument('--data', '-d')
parser.add_argument('--server', '-s')
parser.add_argument('--embeddingsize', '-emb')
parser.add_argument('--layernum', '-layer')
parser.add_argument('--nhead', '-nhead')

cmdargs = parser.parse_args()

usegpu = True

if cmdargs.gpu is None:
    usegpu = False
    args['device'] = 'cpu'
else:
    usegpu = True
    args['device'] = 'cuda:' + str(cmdargs.gpu)

if cmdargs.batch is None:
    pass
else:
    args['batchSize'] = int(cmdargs.batch)

if cmdargs.modelarch is None:
    pass
else:
    args['LMtype'] = cmdargs.modelarch

if cmdargs.data is None:
    pass
else:
    args['corpus'] = cmdargs.data
    args['typename'] = args['corpus']

if cmdargs.server is None:
    args['server'] = 'other'
else:
    args['server'] = cmdargs.server

if cmdargs.embeddingsize is None:
    pass
else:
    args['embeddingSize'] = int(cmdargs.embeddingsize)


if cmdargs.layernum is None:
    pass
else:
    args['numLayers'] = int(cmdargs.layernum)
if cmdargs.nhead is None:
    args['nhead'] = 1
else:
    args['nhead'] = int(cmdargs.nhead)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), datetime.datetime.now())


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens

class Runner:
    def __init__(self):
        self.model_path = args['rootDir'] + '/model.pth'
        # self.model_bw_path = args['rootDir'] + '/model_bw.pth'
        self.drawcount = 0

        self.l1, self.l2 = args['typename'].split('_')


    def main(self):
        self.textData =  TextData_MT('MT', withbpe=False)

        args['maxLengthEnco'] = args['maxLength'] = 200
        args['maxLengthDeco'] =  args['maxLengthEnco'] + 1
        self.start_token = self.textData.word2index['<s>']
        self.end_token = self.textData.word2index['</s>']
        torch.manual_seed(0)



        print(args)




        # self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained('xlnet-base-cased', 'xlnet-base-cased')
        # self.encoder_model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        # self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

        # self.trainMT()     # contain  model saving
        # self.evaluateRandomly()
        self.testMT()

    def convert_to_features(self, example_batch):
        input_encodings = self.tokenizer.batch_encode_plus(example_batch[0], pad_to_max_length=True, max_length=1024,
                                                      truncation=True)
        target_encodings = self.tokenizer.batch_encode_plus(example_batch[1], pad_to_max_length=True,
                                                       max_length=1024, truncation=True)

        labels = target_encodings['input_ids']
        decoder_input_ids = shift_tokens_right(labels, model.config.pad_token_id)
        labels[labels[:, :] == model.config.pad_token_id] = -100

        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
        }

        return encodings

    def trainMT(self, print_every=10000, plot_every=10, learning_rate=0.001):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        print(type(self.textData.word2index), args['device'])

        iter = 1
        batches = self.textData.getBatches()
        batches_dev = self.textData.getBatches('valid')
        n_iters = len(batches)
        print('niters ',n_iters)

        args['trainseq2seq'] = False

        min_BLEU = -1

        self._num_updates = 0
        # self.Cal_BLEU_for_dataset('test')

        CE_loss_total = 0
        KL_total = 0
        VAE_recon_total = 0
        error_total = 0
        print('begin training...')
        epoch = 1

        # while epoch < args['numEpochs']:
        #     losses = []




        encodings = [self.convert_to_features([batch.encoderSeqs, batch.decoderSeqs]) for batch in batches]
        encodings_dev = [self.convert_to_features([batch.encoderSeqs, batch.decoderSeqs]) for batch in batches_dev]
        # sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].to(args['device'])
        # sample['net_input']['src_lengths']= sample['net_input']['src_lengths'].to(args['device'])
        # sample['net_input']['prev_output_tokens'] =  sample['net_input']['prev_output_tokens'].to(args['device'])
        # sample['target'] = sample['target'].to(args['device'])
        training_args = TrainingArguments(
            output_dir='./models/bart-summarizer',
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=encodings,
            eval_dataset=encodings_dev
        )
        trainer.train()


    def testMT(self):
        start = time.time()
        print('Test set BLEU = ', self.Cal_BLEU_for_dataset('test'))
        end = time.time()
        print('Test time: ', time.strftime("%H:%M:%S", time.gmtime(end-start )))

    def Cal_BLEU_for_dataset(self, datasetname, print_test_files = True):
        EVAL_BLEU_ORDER = 4
        if not hasattr(self, 'testbatches' ):
            self.testbatches = {}
        if datasetname not in self.testbatches:
            self.testbatches[datasetname] = self.textData.getBatches(setname=datasetname)


        num = 0
        ave_loss = 0
        pred_ans = []  # bleu
        gold_ans = []
        rec = None
        valid_loss = []

        if print_test_files:
            record_file = open(args['rootDir'] + 'record_test_nobpe.txt', 'w')


        with torch.no_grad():
            # print(len(self.testbatches[datasetname][0].decoderSeqs))
            for batch in self.testbatches[datasetname]:


                # x = {}
                #
                # x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
                # x['dec_input'] = autograd.Variable(torch.LongTensor(batch.decoderSeqs)).to(args['device'])
                # x['dec_len'] = batch.decoder_lens
                # x['target'] = autograd.Variable(torch.LongTensor(batch.targetSeqs)).to(args['device'])
                # print(batch.encoderSeqs )
                inputs = self.tokenizer(batch.encoderSeqs, max_length=2000, return_tensors='pt', padding=True)
                print([len(s.split())for s in batch.encoderSeqs], inputs['input_ids'].size())
                # print(self.tokenizer.vocab_size)
                # print(dir(self.model))
                # Generate Summary
                summary_ids = self.model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
                pred_tokens = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]


                if print_test_files:
                    pred_ans_this = [h.split() for h in pred_tokens]
                    gold_ans_this = [[r.split()] for r in batch.targetSeqs]
                    bleus=[]
                    for pat, gat in zip(pred_ans_this, gold_ans_this):

                        b = corpus_bleu([gat], [pat])
                        bleus.append(b)

                    for i in range(len(logging_output['hyps'])):
                        record_file.write(str(logging_output['score'][i]))
                        record_file.write('\t')
                        record_file.write(str(bleus[i]))
                        record_file.write('\t')
                        record_file.write(logging_output['hyps'][i])
                        record_file.write('\t')
                        record_file.write(logging_output['refs'][i])
                        record_file.write('\n')


                pred_ans.extend([h.split() for h in pred_tokens])
                gold_ans.extend([[r.split()] for r in batch.targetSeqs])
                # valid_loss.append(loss)
                # if rec is None:
                #     rec = (decoded_words[0], batch.raw_source[0], batch.raw_target[0])

            bleu_ori = corpus_bleu(gold_ans, pred_ans)

        # self.model.train()
        # self.criterion.train()
        return bleu_ori#, valid_loss

 

if __name__ == '__main__':
    args['corpus'] = 'DE_EN'
    args['typename'] = args['corpus']
    args['embeddingSize'] = 512
    # args['LMtype'] = 'transformer'
    args['norm_attn'] = True
    r = Runner()
    r.main()