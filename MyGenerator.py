import codecs

import numpy as np

from LEmbedding import LEmbedding
from preprocess import preProcess


class Generator(object):
    def __init__(self, para):
        self._para = para

    def make_batches(self, size, batch_size):
        batch_num = int(np.ceil(size / float(batch_size)))
        return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(batch_num)]

    def bert_generator(self, batch_size, data_path, sep, y, shuffle=True):
        index_array = np.arange(y.shape[0])
        if shuffle:
            np.random.shuffle(index_array)
        _parse_data = preProcess(self._para)._parse_data
        data = _parse_data(codecs.open(data_path, 'r'), sep=sep)
        data = [[item[0] for item in sent] for sent in data]
        batches = self.make_batches(y.shape[0]-1, batch_size)
        le = LEmbedding(self._para)
        while True:
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_idx = index_array[batch_start:batch_end]
                batch_data = [data[idx] for idx in batch_idx]
                batch_x = le.embedding(batch_data)
                batch_x = batch_x[:, 1:self._para.max_len+1]
                batch_y = y[batch_idx]
                yield batch_x, batch_y
