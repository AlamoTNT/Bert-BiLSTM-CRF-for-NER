import codecs
import logging

import keras
import numpy as np
from keras_bert import load_trained_model_from_checkpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LEmbedding(object):
    def __init__(self, bert_config):
        """
        构造预训练类
        :param bert_config: bert预训练模型相关配置
        """
        config_path = bert_config.config_path + '/bert_config.json'
        checkpoint_path = bert_config.config_path + '/bert_model.ckpt'
        model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
        model = keras.models.Model(inputs=model.inputs, outputs=model.output)
        model.summary(line_length=120)
        self._model = model
        self._config = bert_config

    def embedding(self, sentences):
        """
        使用官方预训练好的bert模型来获取句子的向量
        :param texts: 需要转换成向量的句子
        :return:
        """
        result = []
        token_dict = {}
        dict_path = self._config.config_path + '/vocab.txt'
        with codecs.open(dict_path, 'r', 'utf8') as rows:
            for line in rows:
                token = line.strip()
                token_dict[token] = len(token_dict)
        for sentence in sentences:
            sentence.append('[SEP]')
            sentence.insert(0, '[CLS]')
            token_input = np.asarray([[token_dict[token] if token in token_dict else token_dict['[unused1]'] for token
                                       in sentence] + [0] * (512 - len(sentence))])
            seg_input = np.asarray([[0] * len(sentence) + [0] * (512 - len(sentence))])
            predicts = self._model.predict([token_input, seg_input])[0]
            result.append(predicts)
            return np.array(result)


if __name__ == '__main__':
    # emb = LEmbedding({'config_path': 'config'})
    # emb_vec = emb.embedding(['你', '好'], ['世', '界', '和', '平'])
    # logger.info(len(emb_vec[0]), len(emb_vec[1]))
    pass
