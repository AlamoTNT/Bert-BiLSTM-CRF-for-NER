import pickle
import logging
import argparse

import numpy as np
from keras.callbacks import ModelCheckpoint

import ModelLib
from preprocess import preProcess
from MyGenerator import Generator
from utils import ner_accuracy, F1

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Train(object):
    def __init__(self, para):
        self._para = para
        self.logger = logger

    def train_bert_model(self, use_generator=False):
        data_path = self._para.data_path
        _, train_y, _, _, _, val_y, _, tags = pickle.load(open(data_path+'/nlp_ner.pk', 'rb'))
        self._para.tag_num = len(tags)
        model = ModelLib.BERT_MODEL(self._para)
        checkpoint = ModelCheckpoint(data_path+'/bert-bilstm', monitor='val_viterbi_acc', verbose=1,
                                     save_best_only=True, mode='max')
        pre_process = preProcess(self._para)
        if use_generator:
            val_x = pre_process.load_bert_test(data_path+'/dev.txt', self._para.sep)
            model.fit_generator(
                Generator(self._para).bert_generator(self._para.batch_size,
                                                     data_path+'/train.txt',
                                                     self._para.sep,
                                                     train_y, shuffle=True),
                steps_per_epoch=train_y.shape[0] // self._para.batch_size + 1,
                callbacks=[checkpoint],
                validation_data=(val_x, val_y),
                epochs=self._para.EPOCHES,
                verbose=1)
        else:
            train_x, val_x = pre_process.load_bert_train_dev()
            logger.info('%s, %s' % (train_x.shape, train_y.shape))
            logger.info('%s, %s' % (val_x.shape, val_y.shape))
            model.fit(train_x, train_y,
                      batch_size=self._para.batch_size,
                      epochs=self._para.EPOCHES,
                      callbacks=[checkpoint],
                      validation_data=(val_x, val_y),
                      shuffle=True,
                      verbose=1)
        model.save(data_path+'/bert-bilstm')


def train(para):
    # prepr = preProcess(para)
    # prepr.train_test_dev_preprocess()
    logger.info('=======Start training process=======')
    train = Train(para)
    train.train_bert_model(use_generator=False)


def test(para):
    logger.info('=======Start testing process=======')
    data_path = data_path = para.data_path
    _, _, _, test_y, _, _, _, tags = pickle.load(open(data_path + '/nlp_ner.pk', 'rb'))
    para.tag_num = len(tags)
    model = ModelLib.BERT_MODEL(para)
    model.load_weights(filepath=data_path+'/bert-bilstm')
    pre_process = preProcess(para)
    test_x = pre_process.load_bert_test()
    lengths = pre_process.get_lengths(test_y)

    pred_y = model.predict(test_x)

    tag_pred_y = []
    tag_test_y = []
    for i, y in enumerate(pred_y):
        y = [np.argmax(dim) for dim in y]
        # logger.info(lengths[i])
        p_y = y[:lengths[i]]
        # logger.info(p_y)
        t_y = test_y[i][:lengths[i]].flatten()
        # logger.info(t_y)
        p_y = [tags[dim] for dim in p_y]
        t_y = [tags[dim] for dim in t_y]
        tag_pred_y.append(p_y)
        tag_test_y.append(t_y)

    accuracy = ner_accuracy(tag_pred_y, tag_test_y)
    logger.info('Test accuracy %f' % accuracy)
    precision, recall, f1_score = F1(tag_pred_y, tag_test_y)
    logger.info('Precision=%f, Recall=%f, f1-score=%f' % (precision, recall, f1_score))
    return tag_pred_y, tag_test_y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set up configuration')

    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--config_path', default='./config')
    parser.add_argument('--embed_dim', type=int, default=200)
    parser.add_argument('--max_len', type=int, default=142)
    parser.add_argument('--EPOCHES', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sep', default=' ')
    parser.add_argument('--char_dropout', default=0.5)
    parser.add_argument('--rnn_dropout', default=0.5)
    parser.add_argument('--lstm_unit', type=int, default=300)
    parser.add_argument('--tag_num', type=int, default=10)

    # 使得参数创建并生效
    args = parser.parse_args()
    if args.train:
        # 模型训练
        train(args)
    else:
        # 模型测试
        test(args)


