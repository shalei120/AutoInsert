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
# from inverse_square_root_schedule import InverseSquareRootSchedule, InverseSquareRootLRScheduleConfig
import fairseq
print(dir(fairseq))
from fairseq.optim import lr_scheduler

from fairseq import criterions
import utils
from trainer import Trainer
import tasks
# from transformer2 import TransformerModel
# from fairseq import
from argparse import Namespace
# from kenLM import LMEvaluator as LMEr
from omegaconf import dictconfig
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
parser.add_argument('--batch', '-b')
parser.add_argument('--modelarch', '-m')
parser.add_argument('--data', '-d')
parser.add_argument('--server', '-s')
parser.add_argument('--embeddingsize', '-emb')
parser.add_argument('--layernum', '-layer')
parser.add_argument('--nhead', '-nhead')
parser.add_argument('--task', '-t')
parser.add_argument('--retrain', '-r')

cmdargs = parser.parse_args()


if cmdargs.gpu is None:
    args['device'] = 'cpu'
else:
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
if cmdargs.task is None:
    args['task'] = 'easy'
else:
    args['task'] = cmdargs.task
if cmdargs.retrain is None:
    args['retrain'] = True
else:
    args['retrain'] = not (cmdargs.retrain == 'False'  )

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




class Runner:
    def __init__(self):
        self.model_path = args['rootDir'] + '/model.pth'
        # self.model_bw_path = args['rootDir'] + '/model_bw.pth'
        self.drawcount = 0

        self.l1, self.l2 = args['typename'].split('_')

    # def criterion(self, model, sample, reduce=True, sentence_avg=False):
    #     """Compute the loss for the given sample.
    #
    #     Returns a tuple with three elements:
    #     1) the loss
    #     2) the sample size, which is used as the denominator for the gradient
    #     3) logging outputs to display while training
    #     """
    #     # print('here',sample)
    #     loss, data = model(sample)
    #     # loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
    #     sample_size = (
    #         sample["target"].size(0) if sentence_avg else sample["ntokens"]
    #     )
    #     logging_output = {
    #         "loss": loss.data,
    #         "nll_loss": data["nll_loss"],
    #         "ntokens": sample["ntokens"],
    #         "nsentences": sample["target"].size(0),
    #         "sample_size": sample_size,
    #     }
    #     return loss, sample_size, logging_output

    def change_word(self, d):

        d.indices['START_TOKEN']=0
        d.indices['PAD']=1
        d.indices['END_TOKEN']=2
        d.indices['UNK']=3
        d.symbols[0]='START_TOKEN'
        d.symbols[1]='PAD'
        d.symbols[2] = 'END_TOKEN'
        d.symbols[3] ='UNK'
        d.bos_word = 'START_TOKEN'
        d.eos_word = 'END_TOKEN'
        d.pad_word = 'PAD'
        d.unk_word = 'UNK'

    def main(self):
        self.textData =  TextData_MT('MT')

        args['maxLengthEnco'] = args['maxLength'] = 200
        args['maxLengthDeco'] =  args['maxLengthEnco'] + 1
        self.start_token = self.textData.word2index['<s>']
        self.end_token = self.textData.word2index['</s>']
        torch.manual_seed(0)

        self.task = tasks.setup_task(dictconfig.DictConfig(
            {'_name': 'translation', 'data': args['rootDir'] + 'preprocessed', 'source_lang': None,
             'target_lang': None, 'load_alignments': False, 'left_pad_source': True, 'left_pad_target': False,
             'max_source_positions': 10000, 'max_target_positions': 10000, 'upsample_primary': -1,
             'truncate_source': False, 'num_batch_buckets': 0, 'train_subset': 'train', 'dataset_impl': None,
             'required_seq_len_multiple': 1, 'eval_bleu': True,
             'eval_bleu_args': '{"beam":5,"max_len_a":1.2,"max_len_b":10}', 'eval_bleu_detok': 'moses',
             'eval_bleu_detok_args': '{}', 'eval_tokenized_bleu': False, 'eval_bleu_remove_bpe': '@@ ',
             'eval_bleu_print_samples': True}))

        # self.change_word(self.task.src_dict)
        # self.change_word(self.task.tgt_dict)
        # self.textData.word2index = self.task.src_dict.indices
        # self.textData.index2word=  self.task.src_dict.symbols
        # self.task.src_dict.indices = self.textData.word2index
        # self.task.src_dict.symbols = self.textData.index2word
        # self.textData.word2index[args['typename']][self.l2]= self.task.tgt_dict.indices
        # self.textData.index2word[args['typename']][self.l2] =  self.task.tgt_dict.symbols
        args['vocabularySize_src'] = len(self.task.src_dict.symbols)
        args['vocabularySize_tgt'] = len(self.task.tgt_dict.symbols)

        print(args['vocabularySize_src'], args['vocabularySize_tgt'])
        print(args)
        self.task.load_dataset('valid', combine=False, epoch=1)
        self.train_itr = self.get_train_iterator(
            epoch=1, load_dataset=True
        )
            self.model = TranslationModel(self.task.src_dict.indices,
                                          self.task.src_dict.symbols,
                                          self.task.tgt_dict.indices,
                                          self.task.tgt_dict.symbols,).to(args['device'])
            self.sequence_generator = self.model.sequence_generator
            self.criterion = self.task.build_criterion(dictconfig.DictConfig({'_name': 'label_smoothed_cross_entropy', 'label_smoothing': 0.1, 'report_accuracy': False, 'ignore_prefix_size': 0, 'sentence_avg': False}))
            # self.model = torch.load(self.model_path.replace('model', 'model_'+'fw'), map_location=args['device'])
            params = sum([np.prod(p.size()) for p in self.model.parameters()])
            print(params, sum([sys.getsizeof(p.storage()) for p in self.model.parameters()])/1000000, 'm')

        if args['retrain']:
            self.trainMT()     # contain  model saving
        else:
            self.model.load_state_dict(torch.load(self.model_path.replace('model', 'model_'+args['LMtype']+'_'+args['task'] + '_'+str(args['maxLength'])+'_'+str(args['numLayers'])+'_'+str(args['embeddingSize'])), map_location=args['device']))

        # self.evaluateRandomly()
        self.testMT()

    def lr_step_begin_epoch(self, epoch, lr_scheduler):
        """Adjust the learning rate at the beginning of the epoch."""
        lr_scheduler.step_begin_epoch(epoch)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update(lr_scheduler)

    def lr_step(self, epoch, lr_scheduler, val_loss=None):
        """Adjust the learning rate at the end of the epoch."""
        # print(val_loss)
        lr_scheduler.step(epoch, val_loss)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update(lr_scheduler)

    def lr_step_update(self, lr_scheduler):
        """Update the learning rate after each update."""
        new_lr = lr_scheduler.step_update(self.get_num_updates())
        # if isinstance(new_lr, dict):
        #     for k, v in new_lr.items():
        #         metrics.log_scalar(f"lr_{k}", v, weight=0, priority=300)
        #     new_lr = new_lr.get("default", next(iter(new_lr.values())))
        # else:
        #     metrics.log_scalar("lr", new_lr, weight=0, priority=300)
        return new_lr

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        # self.lr_step_update()
        # if self.quantizer:
        #     self.quantizer.step_update(self._num_updates)
        # metrics.log_scalar("num_updates", self._num_updates, weight=0, priority=200)

    def trainMT(self, print_every=10000, plot_every=10, learning_rate=0.001):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        print(type(self.textData.word2index), args['device'])

        # learning_rate =
        # optimizer = optim.Adam(self.model.parameters(), lr=0.001,betas=(0.9, 0.98), eps=1e-08, weight_decay = 0.0001)#, amsgrad=True)
        optimizer = fairseq.optim.build_optimizer(dictconfig.DictConfig(
            {'_name': 'adam', 'adam_betas': '(0.9,0.98)', 'adam_eps': 1e-08, 'weight_decay': 0.0001,
             'use_old_adam': False, 'tpu': False, 'lr': [0.0005]}), list(self.model.parameters()) + list(self.criterion.parameters()))

        # optimizer = torch.optim.SGD(self.model.parameters(), lr=5.0)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        scheduler = lr_scheduler.build_lr_scheduler(
            dictconfig.DictConfig({'_name': 'inverse_sqrt', 'warmup_updates': 4000, 'warmup_init_lr': -1.0, 'lr': [0.0005]}),
            optimizer,
        )
        iter = 1
        batches = self.textData.getBatches()
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


        self.lr_step_begin_epoch(epoch,scheduler)
        while epoch < args['numEpochs']:
            losses = []
            itr = self.train_itr.next_epoch_itr(
                fix_batches_to_gpus=False ,
                shuffle=(self.train_itr.next_epoch_idx > 0),
            )

            # for batch in batches:
            for sample in itr:

                # self._set_seed()
                optimizer.zero_grad()
                # x={}
                # x['id'] = torch.LongTensor(batch.id).to(args['device'])
                # x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
                # x['dec_input'] = autograd.Variable(torch.LongTensor(batch.decoderSeqs)).to(args['device'])
                # x['dec_len'] = batch.decoder_lens
                # x['target'] = autograd.Variable(torch.LongTensor(batch.targetSeqs)).to(args['device'])

                sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].to(args['device'])
                sample['net_input']['src_lengths']= sample['net_input']['src_lengths'].to(args['device'])
                sample['net_input']['prev_output_tokens'] =  sample['net_input']['prev_output_tokens'].to(args['device'])
                sample['target'] = sample['target'].to(args['device'])

                # loss, data = self.model(sample)    # batch seq_len outsize
                # loss, sample_size, logging_output = self.criterion(self.model,sample)

                loss, sample_size_i, logging_output = self.train_step(
                    sample=sample,
                    model=self.model,
                    criterion=self.criterion,
                    optimizer=optimizer,
                    update_num=self.get_num_updates(),
                    ignore_grad=False,
                )
                loss_mean = torch.mean(loss)

                # # # Reward = loss_mean.data
                # #
                # optimizer.backward(loss_mean, retain_graph=True)
                # loss_mean.backward(retain_graph=True)
                #
                # # torch.nn.utils.clip_grad_norm_(self.model.parameters(), args['clip'])
                #
                # optimizer.all_reduce_grads(self.model)
                if torch.is_tensor(sample_size_i):
                    sample_size_i = sample_size_i.float()
                else:
                    sample_size_i = float(sample_size_i)
                optimizer.multiply_grads(1 / (sample_size_i or 1.0))
                # grad_norm = self.clip_grad_norm(0.0)
                # # self._check_grad_norms(grad_norm)
                optimizer.step()

                # log_output = trainer.train_step([sample])
                # loss_mean = log_output['loss']

                print_loss_total += loss_mean
                plot_loss_total += loss_mean

                losses.append(loss_mean)
                self.set_num_updates(self.get_num_updates() + 1)
                self.lr_step_update(scheduler)
                # GPUtil.showUtilization()
                if iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    KL_avg = KL_total / print_every
                    KL_total = 0
                    CEloss_avg = CE_loss_total / print_every
                    CE_loss_total = 0
                    VAE_recon_avg = VAE_recon_total / print_every
                    VAE_recon_total = 0
                    error_avg = error_total / print_every
                    error_total = 0
                    print('%s (%d %d%%) loss = %.4f, CE_loss = %.4f, VAE recon = %.4f, KL = %.4f, error=%.4f' % (timeSince(start, iter / n_iters),
                                                 iter, iter / n_iters * 100, print_loss_avg, CEloss_avg, VAE_recon_avg, KL_avg, error_avg))
                    # GPUtil.showUtilization()
                    # del de_output,loss,true_mean
                    # GPUtil.showUtilization()
                    # torch.cuda.empty_cache()
                    # GPUtil.showUtilization()
                    if args['corpus'] != '1mb' or iter % 100000 == 0:
                        BLEU, bleu_ori,val_losses = self.Cal_BLEU_for_dataset('test')

                        print('Test ppl: ', BLEU, bleu_ori, 'Val loss: ', sum(val_losses)/ len(val_losses))

                        if BLEU > min_BLEU or min_BLEU == -1:
                            print('BLEU = ', BLEU, bleu_ori, '>= min_BLEU (', min_BLEU, '), saving model...')
                            min_BLEU = BLEU

                if iter % plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0

                iter+=1

            # scheduler.step()
            #     for g in optimizer.param_groups:
            #         g['lr'] = args['embeddingSize'] ** (-0.5) * min(iter ** (-0.5), iter*(4000**(-1.5)))
            BLEU, bleu_ori, valid_losses = self.Cal_BLEU_for_dataset('test')
            if BLEU > min_BLEU or min_BLEU == -1:
                print('BLEU = ', BLEU, bleu_ori, '>= min_BLEU (', min_BLEU, '), saving model...')
                torch.save(self.model.state_dict(), self.model_path.replace('model', 'model_'+args['LMtype']+'_'+args['task'] + '_'+str(args['maxLength'])+'_'+str(args['numLayers'])+'_'+str(args['embeddingSize'])))
                min_BLEU = BLEU
            lr = self.lr_step(epoch, scheduler,valid_losses[0])
            print('Epoch ', epoch, 'loss = ', sum(losses) / len(losses), 'Valid BLEU = ', BLEU, bleu_ori,'best BLEU: ', min_BLEU)
            epoch += 1

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

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
        self.model.eval()
        # self.criterion.eval()
        num = 0
        ave_loss = 0
        pred_ans = []  # bleu
        gold_ans = []
        rec = None
        valid_loss = []

        if print_test_files:
            record_file = open(args['rootDir'] + 'record_test.txt', 'w')

        valid_epoch_itr = self.get_valid_iterator('valid')
        print(type(valid_epoch_itr),dir(valid_epoch_itr))
        itr = valid_epoch_itr.next_epoch_itr (
            shuffle=False#, set_dataset_epoch=False  # use a fixed valid set
        )

        _bleu_counts_ = 0
        _bleu_totals_ = 0
        _bleu_sys_len = 0
        _bleu_ref_len = 0
        n=0
        with torch.no_grad():
            # print(len(self.testbatches[datasetname][0].decoderSeqs))
            # for batch in self.testbatches[datasetname]:z
            for sample in itr:

                # x = {}
                #
                # x['id'] = torch.LongTensor(batch.id).to(args['device'])
                # x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
                # x['dec_input'] = autograd.Variable(torch.LongTensor(batch.decoderSeqs)).to(args['device'])
                # x['dec_len'] = batch.decoder_lens
                # x['target'] = autograd.Variable(torch.LongTensor(batch.targetSeqs)).to(args['device'])
                sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].to(args['device'])
                sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].to(args['device'])
                sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].to(args['device'])
                sample['target'] = sample['target'].to(args['device'])
                loss, sample_size, logging_output = self.model.predict(sample)    # batch seq_len outsize
                # loss, sample_size, logging_output = self.task.valid_step(
                #     sample, self.model, self.criterion
                # )
                if print_test_files:
                    pred_ans_this = [h.split() for h in logging_output['hyps']]
                    gold_ans_this = [[r.split()] for r in logging_output['refs']]
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


                pred_ans.extend([h.split() for h in logging_output['hyps']])
                gold_ans.extend([[r.split()] for r in logging_output['refs']])
                valid_loss.append(loss)
                # if rec is None:
                #     rec = (decoded_words[0], batch.raw_source[0], batch.raw_target[0])


                counts, totals = [], []
                for i in range(EVAL_BLEU_ORDER):
                    counts.append(logging_output["_bleu_counts_" + str(i)])
                    totals.append(logging_output["_bleu_totals_" + str(i)])
                _bleu_counts_ = (_bleu_counts_ * n + np.array(counts))/ (n+1)
                _bleu_totals_ = (_bleu_totals_ * n + np.array(totals))/ (n+1)
                _bleu_sys_len = (_bleu_sys_len * n + logging_output["_bleu_sys_len"])/ (n+1)
                _bleu_ref_len = (_bleu_ref_len * n + logging_output["_bleu_ref_len"])/ (n+1)
                n+=1
            metrics = {}
            metrics["_bleu_counts"] = _bleu_counts_
            metrics["_bleu_totals"]=_bleu_totals_
            metrics["_bleu_sys_len"]=_bleu_sys_len
            metrics["_bleu_ref_len"]=_bleu_ref_len

            print(metrics)

            def compute_bleu(meters):
                import inspect
                import sacrebleu

                fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                if "smooth_method" in fn_sig:
                    smooth = {"smooth_method": "exp"}
                else:
                    smooth = {"smooth": "exp"}
                bleu = sacrebleu.compute_bleu(
                    correct=np.array(meters["_bleu_counts"]),
                    total=np.array(meters["_bleu_totals"]),
                    sys_len=meters["_bleu_sys_len"],
                    ref_len=meters["_bleu_ref_len"],
                    **smooth
                )
                return round(bleu.score, 2)

            metrics["bleu"] = compute_bleu(metrics)
            print(rec)
            bleu = metrics["bleu"]
            bleu_ori = corpus_bleu(gold_ans, pred_ans)

        self.model.train()
        self.criterion.train()
        return bleu, bleu_ori, valid_loss

    def get_train_iterator(
        self,
        epoch,
        combine=True,
        load_dataset=True,
        data_selector=None,
        shard_batch_itr=True,
        disable_iterator_cache=False,
    ):
        self.data_parallel_world_size = 1
        self.data_parallel_rank = 0
        """Return an EpochBatchIterator over the training set for a given epoch."""
        if load_dataset:
            print("loading train data for epoch {}".format(epoch))
            self.task.load_dataset(
                'train',
                epoch=epoch,
                combine=combine,
                data_selector=data_selector,
                tpu=False,
            )
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.dataset('train'),
            max_tokens=4096,
            max_sentences=None,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                # self.model.max_positions(),
                4096,
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=8,
                seed=1,
            num_shards=self.data_parallel_world_size if shard_batch_itr else 1,
            shard_id=self.data_parallel_rank if shard_batch_itr else 0,
                num_workers=1,
            epoch=epoch,
            data_buffer_size=10,
            disable_iterator_cache=disable_iterator_cache,
        )
        # self.reset_dummy_batch(batch_iterator.first_batch)
        return batch_iterator

    def get_valid_iterator(
        self,
        subset,
        disable_iterator_cache=False,
    ):
        self.data_parallel_world_size = 1
        self.data_parallel_rank = 0
        """Return an EpochBatchIterator over given validation subset for a given epoch."""
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.dataset(subset),
            max_tokens=10000,
            max_sentences=None,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                # self.model.max_positions(),
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=8,
            seed=1,
            num_shards=self.data_parallel_world_size,
            shard_id=self.data_parallel_rank,
            num_workers=1,
            # always pass a fixed "epoch" to keep validation data consistent
            # across training epochs
            epoch=1,
            data_buffer_size=10,
            disable_iterator_cache=disable_iterator_cache,
        )
        # self.reset_dummy_batch(batch_iterator.first_batch)
        return batch_iterator
 

if __name__ == '__main__':
    args['corpus'] = 'DE_EN'
    args['typename'] = args['corpus']
    args['embeddingSize'] = 512
    # args['LMtype'] = 'transformer'
    args['norm_attn'] = True
    r = Runner()
    r.main()