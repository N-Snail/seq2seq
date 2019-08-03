# -*- coding: utf-8 -*-

import tensorflow as tf
from data_helper import load_dict,loadDataset, getBatches, sentence2enco
from seq2seq_model import Seq2SeqModel
import sys
import numpy as np

padId, unkId, eosId, goId = 0, 1, 2, 3

tf.app.flags.DEFINE_integer('rnn_size', 256, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 300, 'Embedding dimensions of encoder and decoder inputs')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.99, 'Learning rate decay factor rate')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_integer('numEpochs', 25, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 100, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_string('model_dir', 'seq2seq_model/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'seq2seq.ckpt', 'File name used for model checkpoints')
FLAGS = tf.app.flags.FLAGS

word2idx, idx2word = load_dict()


def predict_ids_to_seq(predict_ids, id2word, beam_szie):
    '''
    将beam_search返回的结果转化为字符串
    :param predict_ids: 列表，长度为batch_size，每个元素都是decode_len*beam_size的数组
    :param id2word: vocab字典
    :return:
    '''
    for single_predict in predict_ids:
        for i in range(beam_szie):
            predict_list = np.ndarray.tolist(single_predict[:, :, i])
            predict_seq = []
            for idx in predict_list[0]:
                if idx != eosId:
                    predict_seq.append(id2word[idx])
                else:
                    break
            print(''.join(predict_seq))


with tf.Session() as sess:
    model = Seq2SeqModel(FLAGS.rnn_size, FLAGS.num_layers, FLAGS.embedding_size, FLAGS.learning_rate,
                         FLAGS.learning_rate_decay_factor, word2idx,
                         mode='decode', use_attention=True, beam_search=True, beam_size=5, max_gradient_norm=5.0)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('No such file:[{}]'.format(FLAGS.model_dir))
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
        batch = sentence2enco(sentence, word2idx)
        predicted_ids = model.infer(sess, batch)
        predict_ids_to_seq(predicted_ids, idx2word, 5)
        print("> ", "")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
