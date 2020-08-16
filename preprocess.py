import pickle
import codecs
import logging
from collections import Counter

import numpy as np
from keras.preprocessing.sequence import pad_sequences

from LEmbedding import LEmbedding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class preProcess(object):
    def __init__(self, para):
        self._para = para
        self.logger = logger

    def _parse_data(self, file_input, sep=' '):
        rows = file_input.readlines()
        rows[0] = rows[0].replace('\xef\xbb\xbf', '')
        items = [row.strip().split(sep) for row in rows]
        sents = []
        sent = []
        n = 0
        for item in items:
            if item.__len__() != 1:
                sent.append(item)
            else:
                if sent.__len__() > self._para.max_len:
                    n += 1
                    split_sent = []
                    for i, item in enumerate(sent):
                        if item[0] in ['。', ',', '，', '！', '!', '?', '？', '、', '：'] and split_sent.__len__() > 50:
                            split_sent.append(item)
                            if split_sent.__len__() < self._para.max_len:
                                sents.append(split_sent[:])
                            split_sent = []
                        else:
                            split_sent.append(item)
                else:
                    if sent.__len__() > 1:
                        sents.append(sent[:])
                sent = []
        self.logger.info('Over max-len sentence num %d' % n)
        return sents

    def get_tag(self, data):
        tag = []
        for words in data:
            for word_tag in words:
                if word_tag[1] not in tag:
                    tag.append(word_tag[1])
        return tag

    def train_test_dev_preprocess(self):
        data_path = self._para.data_path
        train = self._parse_data(codecs.open(data_path+'/train.txt', 'r', encoding='utf8'), sep=self._para.sep)
        test = self._parse_data(codecs.open(data_path+'/test.txt', 'r', encoding='utf8'), sep=self._para.sep)
        dev = self._parse_data(codecs.open(data_path+'/dev.txt', 'r', encoding='utf8'), sep=self._para.sep)
        self.logger.info('Load train-test-dev dataset finished!')
        dataset = train + test + dev
        tags = self.get_tag(dataset)
        self.logger.info(tags)
        self.logger.info('(%d, %d, %d, %d)' % (train.__len__(), test.__len__(), dev.__len__(), dataset.__len__()))
        # 词频统计
        word_counts = Counter(row[0].lower() for sample in dataset for row in sample)
        # 词典
        vocab = [w for w, f in iter(word_counts.items()) if f >= 1]
        # 保留id=[0, 1]作为[PAD, UNK]
        word2id = dict((w, i+2) for i, w in enumerate(vocab))

        train_X, train_Y = self.process_data(train, word2id, tags)
        test_X, test_Y = self.process_data(test, word2id, tags)
        dev_X, dev_Y = self.process_data(dev, word2id, tags)
        pickle.dump((train_X, train_Y, test_X, test_Y, dev_X, dev_Y, word2id, tags),
                     open(data_path+'/nlp_ner.pk', 'wb'))

    def process_data(self, data, word2idx, chunk_tags, onehot = False):
        x = [[word2idx.get(w[0], 1) for w in s] for s in data]
        y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]
        x = pad_sequences(x, self._para.max_len, padding='post', truncating='post')
        y_chunk = pad_sequences(y_chunk, self._para.max_len, value=-1, padding='post', truncating='post')

        if onehot:
            y_chunk = np.eye(len(chunk_tags), dtype='float32')[y_chunk]
        else:
            y_chunk = np.expand_dims(y_chunk, 2)
        return x, y_chunk

    def load_bert_train_dev(self):
        data_path = self._para.data_path
        train = self._parse_data(codecs.open(data_path+'/train.txt', 'r', encoding='utf-8'), sep=self._para.sep)
        dev = self._parse_data(codecs.open(data_path+'/dev.txt', 'r', encoding='utf-8'), sep=self._para.sep)
        train = [[item[0] for item in sent] for sent in train]
        dev = [[item[0] for item in sent] for sent in dev]

        train_x = np.zeros(shape=(train.__len__(), self._para.max_len, 768), dtype='float32')
        dev_x = np.zeros(shape=(dev.__len__(), self._para.max_len, 768), dtype='float32')
        step = train.__len__() // 256 + 1
        self.logger.info('Train step %d' % step)
        le = LEmbedding(self._para)
        for i in range(step):
            if i != step-1:
                x = le.embedding(train[i*256:(i+1)*256])
                x = x[:, 0:self._para.max_len]
                train_x[i*256:(i+1)*256] = x
            else:
                x = le.embedding(train[i*256:])
                x = x[:, 0:self._para.max_len]
                train_x[i*256:] = x

        step = dev.__len__() // 256 + 1
        self.logger.info('Valid step %d' % step)
        for i in range(step):
            if i != step-1:
                x = le.embedding(dev[i*256:(i+1)*256])
                x = x[:, 0:self._para.max_len]
                dev_x[i*256:(i+1)*256] = x
            else:
                x = le.embedding(dev[i*256:])
                x = x[:, 0:self._para.max_len]
                dev_x[i*256:] = x
        return train_x, dev_x

    def load_bert_test(self):
        data_path = self._para.data_path
        test = self._parse_data(codecs.open(data_path + '/test.txt', 'r', encoding='utf-8'), sep=self._para.sep)
        test = [[item[0] for item in sent] for sent in test]

        test_x = np.zeros(shape=(test.__len__(), self._para.max_len, 768), dtype='float32')
        step = test.__len__() // 256 + 1
        self.logger.info('Test step %d' % step)
        le = LEmbedding(self._para)
        for i in range(step):
            if i != step-1:
                x = le.embedding(test[i*256:(i+1)*256])
                x = x[:, 0:self._para.max_len]
                test_x[i*256:(i+1)*256] = x
            else:
                x = le.embedding(test[i*256:])
                x = x[:, 0:self._para.max_len]
                test_x[i*256:] = x
        return test_x

    def get_lengths(self, X):
        lengths = []
        for i in range(len(X)):
            length = 0
            for dim in X[i]:
                if dim != 0:
                    length += 1
                else:
                    break
            lengths.append(length)
        return lengths


if __name__ == '__main__':
    pass