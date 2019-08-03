# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import nltk
import numpy as np
import pickle
import json
import random
import jieba
import re


padId, unkId, eosId, goId = 0, 1, 2, 3

_DIGIT_RE = re.compile("\d")

class Batch:
    # batch类，里面包含了encoder输入，decoder输入，decoder标签，decoder样本长度mask
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []
        self.emo_cat = []


def loadDataset(filename):
    '''
    读取样本数据
    :param filename: 文件路径，是一个字典，包含word2id、id2word分别是单词与索引对应的字典和反序字典，
                    trainingSamples样本数据，每一条都是QA对
    :return: trainingSamples
    '''
    dataset_path = os.path.join(filename)
    print('Loading dataset from {}'.format(dataset_path))
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
    return data


def createBatch(samples):
    '''
    根据给出的samples（就是一个batch的数据），进行padding并构造成placeholder所需要的数据形式
    :param samples: 一个batch的样本数据，列表，每个元素都是[question， answer，emo_cat]的形式，id
    :return: 处理完之后可以直接传入feed_dict的数据格式
    '''
    batch = Batch()
    batch.encoder_inputs_length = [len(sample[0]) for sample in samples]
    batch.decoder_targets_length = [len(sample[1])+1 for sample in samples]

    max_source_length = max(batch.encoder_inputs_length)
    max_target_length = max(batch.decoder_targets_length)

    for sample in samples:
        source = sample[0]
        pad = [padId] * (max_source_length - len(source))
        batch.encoder_inputs.append(source+pad)

        # 将target进行PAD，并添加END符号
        target = sample[1] + [eosId]
        pad = [padId] * (max_target_length - len(target))
        batch.decoder_targets.append(target + pad)

        batch.emo_cat.append(sample[2])

    # print('batch.encoder_inputs',batch.encoder_inputs)
    # print('batch.encoder_inputs_length',batch.encoder_inputs_length)
    # print('batch.decoder_targets',batch.decoder_targets)
    # print('batch.decoder_targets_length',batch.decoder_targets_length)
    # print('batch.emo_cat',batch.emo_cat)
    return batch


def getBatches(data, batch_size):
    '''
    根据读取出来的所有数据和batch_size将原始数据分成不同的小batch。对每个batch索引的样本调用createBatch函数进行处理
    :param data: loadDataset函数读取之后的trainingSamples
    :param batch_size: batch大小
    :param en_de_seq_len: 列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列的最大长度
    :return: 列表，每个元素都是一个batch的样本数据，可直接传入feed_dict进行训练
    '''
    # 每个epoch之前都要进行样本的shuffle
    random.shuffle(data)
    batches = []
    data_len = len(data)

    def genNextSamples():
        for i in range(0, data_len, batch_size):
            yield data[i:min(i + batch_size, data_len)]

    for samples in genNextSamples():
        batch = createBatch(samples)
        batches.append(batch)
    return batches


def sentence2enco(sentence, word2id):
    '''
    测试的时候将用户输入的句子转化为可以直接feed进模型的数据，现将句子转化成id，然后调用createBatch处理
    :param sentence: 用户输入的句子
    :param word2id: 单词与id之间的对应关系字典
    :param en_de_seq_len: 列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列的最大长度
    :return: 处理之后的数据，可直接feed进模型进行预测
    '''
    if sentence == '':
        return None
    # 分词
    cut_words = list(jieba.cut(sentence))
    # tokens = nltk.word_tokenize(sentence)
    if len(cut_words) > 25:
        cut_words = cut_words[:25]
    # 将每个单词转化为id
    wordIds = []
    for w in cut_words:
        token = _DIGIT_RE.sub("0", w)
        if token in word2id:
            wordIds.append(word2id[token])
        else:
            wordIds.append(unkId)
    # 调用createBatch构造batch
    # print(wordIds)
    batch = createBatch([[wordIds, [],[]]])
    return batch


def data2idx(word2idx):
    json_file = open('../data/new_train_data1.json', 'r')  # '../data/new_test_data1.json'
    data = json.load(json_file)

    data_idx = []

    for pairs in data:
        p_post = []
        for w in pairs[0][0].strip().split(' '):
            word = _DIGIT_RE.sub("0", w)
            if word in word2idx:
                p_post.append(word2idx[word])
            else:
                p_post.append(word2idx['_UNK'])
        # post_emo=pairs[0][1]

        p_response = []
        for w in pairs[1][0].strip().split(' '):
            word = _DIGIT_RE.sub("0", w)
            if word in word2idx:
                p_response.append(word2idx[word])
            else:
                p_response.append(word2idx['_UNK'])
        response_emo = pairs[1][1]

        new_pairs = [p_post, p_response, response_emo]
        data_idx.append(new_pairs)

    file = open('../data/train_new_data_idx.pkl', 'wb')  # '../data/test_data_idx.pkl'
    pickle.dump(data_idx, file)
    # print(len(data))  # 946146
    # print(data[0])
    # print(data_idx[0])


def load_dict():
    dict_file = open('../data/vocabulary_size_40000', 'r', encoding='utf-8').readlines()
    word2idx = {}
    idx2word = []
    for w in dict_file:
        w = w.strip()
        idx2word.append(w)
        word2idx[w] = len(word2idx)
    # print(word2idx)
    # print(idx2word)
    return word2idx, idx2word


# if __name__ == '__main__':
#     word2idx, idx2word = load_dict()
#     data2idx(word2idx)
# file = open('../data/train_new_data_idx.pkl', 'rb')
# data = pickle.load(file)
# print(data,'\n')
# json_file = open('../data/new_train_data1.json', 'rb')  # '../data/new_test_data1.json'
# data2 = json.load(json_file)
# print(data2)