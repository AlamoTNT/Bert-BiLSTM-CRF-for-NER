from keras.models import Model
from keras.layers import Bidirectional, LSTM
from keras.layers import Input
from keras.layers.core import *
from keras_contrib import losses, metrics
from keras_contrib.layers import CRF


def BERT_MODEL(para):
    bert_input = Input(shape=(para.max_len, 768,), dtype='float32', name='bert_input')
    mask = Masking()(bert_input)
    repre = Dropout(para.char_dropout)(mask)
    repre = Dense(300, activation='relu')(repre)
    repre = Bidirectional(LSTM(para.lstm_unit, return_sequences=True, dropout=para.rnn_dropout))(repre)
    crf = CRF(para.tag_num, sparse_target=True)
    crf_output = crf(repre)
    model = Model(inputs=bert_input, outputs=crf_output)
    model.summary()
    model.compile('adam', loss=losses.crf_loss, metrics=[metrics.crf_accuracy])
    return model


if __name__ == '__main__':
    pass