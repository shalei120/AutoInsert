
import numpy as np
import nltk  # For tokenize
nltk.download('punkt')
from nltk.probability import FreqDist
from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance
import random, json
import string, copy
from nltk.tokenize import word_tokenize
from Hyperparameters_MT import args
import requests, tarfile
from learn_bpe import learn_bpe
from apply_bpe import BPE
from bs4 import BeautifulSoup
import codecs
class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        self.encoderSeqs = []
        self.label = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.encoder_lens = []
        self.decoder_lens = []
        self.raw_source = []
        self.raw_target = []


class TextData_MT:
    """Dataset class
    Warning: No vocabulary limit
    """


    def __init__(self, corpusname, taskID = 1):
        """Load all conversations
        Args:
            args: parameters of the model
        """
        self.taskID = taskID
        self.bpe_tmp_filename = args['rootDir'] + 'bpe_tmp.txt'
        # Path variables
        self.tokenizer = word_tokenize

        if not os.path.exists(args['rootDir']):
            os.mkdir(args['rootDir'])

        self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]

        self.word2index = {}
        self.index2word = {}  # For a rapid conversion (Warning: If replace dict by list, modify the filtering to avoid linear complexity with del)

        self.loadCorpus(corpusname)

        self.write_fairseq_dataset()


    def _printStats(self):
        print('Loaded {}: {} words, {} QA'.format('LMbenchmark', len(self.word2index), len(self.trainingSamples)))


    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.datasets['train'])

    def _createBatch(self, samples):
        """Create a single batch from the list of sample. The batch size is automatically defined by the number of
        samples given.
        The inputs should already be inverted. The target should already have <go> and <eos>
        Warning: This function should not make direct calls to args['batchSize'] !!!
        Args:
            samples (list<Obj>): a list of samples, each sample being on the form [input, target]
        Return:
            Batch: a batch object en
        """

        batch = Batch()
        batchSize = len(samples)

        maxlen_def = args['maxLengthEnco'] #if setname == 'train' else 511

        # Create the batch tensor
        for i in range(batchSize):
            # Unpack the sample
            src_sample, add, tgt_sample, raw_src, raw_add, raw_tgt = samples[i]

            if len(src_sample) > maxlen_def:
                src_sample = src_sample[:maxlen_def]
            if len(tgt_sample) > maxlen_def:
                tgt_sample = tgt_sample[:maxlen_def]

            batch.encoderSeqs.append(src_sample)
            batch.decoderSeqs.append([self.word2index['<s>']] + tgt_sample)  # Add the <go> and <eos> tokens
            batch.targetSeqs.append(tgt_sample + [self.word2index['</s>']])  # Same as decoder, but shifted to the left (ignore the <go>)

            assert len(batch.decoderSeqs[i]) <= maxlen_def +1

            # TODO: Should use tf batch function to automatically add padding and batch samples
            # Add padding & define weight
            batch.encoder_lens.append(len(batch.encoderSeqs[i]))
            batch.decoder_lens.append(len(batch.targetSeqs[i]))
            batch.raw_source.append(raw_src)
            batch.raw_target.append(raw_tgt)

        maxlen_dec = max(batch.decoder_lens)
        maxlen_enc = max(batch.encoder_lens)


        for i in range(batchSize):
            batch.encoderSeqs[i] = batch.encoderSeqs[i] + [self.word2index['<pad>']] * (maxlen_enc - len(batch.encoderSeqs[i]))
            batch.decoderSeqs[i] = batch.decoderSeqs[i] + [self.word2index['<pad>']] * (maxlen_dec - len(batch.decoderSeqs[i]))
            batch.targetSeqs[i]  = batch.targetSeqs[i]  + [self.word2index['<pad>']] * (maxlen_dec - len(batch.targetSeqs[i]))

        # pre_sort_list = [(a, b, c) for a, b, c  in
        #                  zip( batch.decoderSeqs, batch.decoder_lens,
        #                      batch.targetSeqs)]
        #
        # post_sorted_list = sorted(pre_sort_list, key=lambda x: x[1], reverse=True)
        #
        # batch.decoderSeqs = [a[0] for a in post_sorted_list]
        # batch.decoder_lens = [a[1] for a in post_sorted_list]
        # batch.targetSeqs = [a[2] for a in post_sorted_list]

        return batch

    def getBatches(self,setname = 'train'):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        self.shuffle()


        batches = []
        batch_size = args['batchSize'] #if setname == 'train' else 32
        print(len(self.datasets[setname]), setname, batch_size)
        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, self.getSampleSize(setname), batch_size):
                yield self.datasets[setname][i:min(i + batch_size, self.getSampleSize(setname))]

        # TODO: Should replace that by generator (better: by tf.queue)

        for index, samples in enumerate(genNextSamples()):
            # print([self.index2word[id] for id in samples[5][0]], samples[5][2])
            batch = self._createBatch(samples)
            batches.append(batch)

        # print([self.index2word[id] for id in batches[2].encoderSeqs[5]], batches[2].raws[5])
        return batches

    def getSampleSize(self,setname):
        """Return the size of the dataset
        Return:
            int: Number of training samples
        """
        return len(self.datasets[setname])

    def getVocabularySize(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.word2index)

    def extract(self, tar_url, extract_path='.'):
        print(tar_url)
        tar = tarfile.open(tar_url, 'r')
        for item in tar:
            tar.extract(item, extract_path)

    def loadCorpus(self, corpusname):
        """Load/create the conversations data
        """
        if args['server'] == 'dgx':
            self.basedir = './.data/'
        else:
            self.basedir = '../'

        if not os.path.exists(self.basedir):
            os.mkdir(self.basedir)

        self.corpusDir = self.basedir + 'dataset-v1/'
        self.fullSamplesPath = args['rootDir'] + '/autoinsert.pkl'  # Full sentences length/vocab

        print(self.fullSamplesPath)
        datasetExist = os.path.isfile(self.fullSamplesPath)
        if not datasetExist:  # First time we load the database: creating all files
            print('Training data not found. Creating dataset...')
            files = os.listdir(self.corpusDir)

            total_words = []

            data_input_seqs = []
            data_add_seqs = []
            data_target_seqs = []
            files = [f for f in files if 'json'  in f]
            files = sorted(files, key=lambda x:int(x.split('.')[0]))
            for filename in files:
                print(filename)

                input_seq = ''
                add_seqs = []
                target_seq = ''
                with open(self.corpusDir +filename, 'r') as src_handle:
                    data = json.load(src_handle)

                    for para in data['paragraphs']:
                        for subsen_dict in para:
                            if subsen_dict['type'] == '=' or subsen_dict['type'] == '-' :
                                input_seq += subsen_dict['text'].strip() + ' '
                            elif subsen_dict['type'] == '+':
                                add_seqs.append(subsen_dict['text'].strip() )

                            if subsen_dict['type'] == '=' or subsen_dict['type'] == '+':
                                target_seq += subsen_dict['text'].strip()  + ' '
                input_seq = ' '.join(self.tokenizer(input_seq))
                for i in range(len(add_seqs)):
                    add_seqs[i] = ' '.join(self.tokenizer(add_seqs[i]))
                target_seq = ' '.join(self.tokenizer(target_seq))
                data_input_seqs.append(input_seq)
                data_add_seqs.append(add_seqs)
                data_target_seqs.append(target_seq)

            self.write_in_tmp_files(data_input_seqs, data_add_seqs, data_target_seqs)

            learn_bpe([self.bpe_tmp_filename], args['rootDir'] + 'auto_insert.bpe', 37000, 6, True)
            codes = codecs.open(args['rootDir'] + 'auto_insert.bpe', encoding='utf-8')
            bpe = BPE(codes, separator='@@')

            data=[]

            for input,add,target in zip(data_input_seqs, data_add_seqs, data_target_seqs):

                input = bpe.process_line(input).split()
                for i in range(len(add)):
                    add[i] = bpe.process_line(add[i]).split()
                    total_words.extend(add[i])
                target = bpe.process_line(target).split()
                total_words.extend(input)
                total_words.extend(target)
                data.append((input,add,target))


            fdist = nltk.FreqDist(total_words)
            sort_count = fdist.most_common(36000)
            print('sort_count: ', len(sort_count))
            with open(args['rootDir'] + "/autoinsert_voc.txt", "w") as v:
                for w, c in tqdm(sort_count):
                    if w not in [' ', '', '\n']:
                        v.write(w)
                        v.write(' ')
                        v.write(str(c))
                        v.write('\n')

                v.close()
            os.system('cp ' + args['rootDir'] + "/autoinsert_voc.txt "+args['rootDir'] + "/fsdata/dict.en.txt")
            os.system('cp ' + args['rootDir'] + "/autoinsert_voc.txt "+args['rootDir'] + "/fsdata/dict.de.txt")

            self.word2index = self.read_word2vec(args['rootDir'] + "/autoinsert_voc.txt")
            self.sorted_word_index = sorted(self.word2index.items(), key=lambda item: item[1])
            print('sorted')
            self.index2word = [w for w, n in self.sorted_word_index]
            print('index2word')
            self.index2word_set = set(self.index2word)
            print('set')

            # self.raw_sentences = copy.deepcopy(dataset)
            random.shuffle(data)
            datanum = len(data)
            dataset={}
            dataset['train'] = data[:int(0.8 * datanum)]
            dataset['valid'] = data[int(0.8 * datanum):int(0.9 * datanum)]
            dataset['test'] = data[int(0.9 * datanum):]
            for setname in ['train', 'valid', 'test']:
                dataset[setname] = [(self.TurnWordID(src),[self.TurnWordID(s) for s in add] ,self.TurnWordID(tgt), src, add, tgt) for src,add, tgt in tqdm(dataset[setname])]
            self.datasets = dataset


            # Saving
            print('Saving dataset...')
            self.saveDataset(self.fullSamplesPath)  # Saving tf samples
        else:
            self.loadDataset(self.fullSamplesPath)

    def write_in_tmp_files(self, data_input_seqs, data_add_seqs, data_target_seqs):
        with open(self.bpe_tmp_filename, 'w') as h:
            for input,add,target in zip(data_input_seqs, data_add_seqs, data_target_seqs):
                h.write(input)
                h.write('\n')
                for s in add:
                    h.write(s)
                    h.write('\n')
                h.write(target)
                h.write('\n')



    def saveDataset(self, filename):
        """Save samples to file
        Args:
            filename (str): pickle filename
        """
        with open(os.path.join(filename), 'wb') as handle:
            data = {'word2index': self.word2index,
                    'index2word': self.index2word,
                    'datasets': self.datasets
                }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def loadDataset(self, filename):
        """Load samples from file
        Args:
            filename (str): pickle filename
        """
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))

        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.word2index = data['word2index']
            self.index2word = data['index2word']
            self.datasets = data['datasets']

    def read_word2vec(self, vocfile ):
        word2index = dict()
        word2index['<s>'] = 0
        word2index['<pad>'] = 1
        word2index['</s>'] = 2
        word2index['<unk>'] = 3
        # word2index['<sep>'] = 4
        cnt = 4
        with open(vocfile, "r") as v:

            for line in v:
                word = line.strip().split()[0]
                word2index[word] = cnt
                print(word,cnt)
                cnt += 1

        print(len(word2index),cnt)
        # dic = {w:numpy.random.normal(size=[int(sys.argv[1])]).astype('float32') for w in word2index}
        print ('Dictionary Got!')
        return word2index

    def TurnWordID(self, words):
        res = []
        for w in words:
            w = w.lower()
            if w in self.index2word_set:
                id = self.word2index[w]
                # if id > 20000:
                #     print('id>20000:', w,id)
                res.append(id)
            else:
                res.append(self.word2index['<unk>'])
        return res

    def write_fairseq_dataset(self):
        folder = args['rootDir'] + '/fsdata/'
        if not os.path.exists(folder):
            os.mkdir(folder)
        src_len = []
        tgt_len = []
        for setname in ['train', 'valid', 'test']:
            with open(folder + setname + '.' + 'de', 'w') as src_h:
                with open(folder + setname + '.' + 'en', 'w') as tgt_h:
                     for _,_,_,src, add, tgt in tqdm(self.datasets[setname]):
                        str_src = src[:1000]
                        for s in add:
                            str_src +=  ['<s>'] +  s
                        str_tgt = tgt[:1000]
                        src_h.write(' '.join(str_src) + '\n')
                        tgt_h.write(' '.join(str_tgt) + '\n')
                        src_len.append(len(str_src))
                        tgt_len.append(len(str_tgt))

        # cfdist = FreqDist(src_len)
        # cfdist.plot()
        print(src_len)
        print(tgt_len)


if __name__ == '__main__':
    args['server'] = 'dgx'
    TextData_MT('MT')