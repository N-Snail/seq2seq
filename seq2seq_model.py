# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib import seq2seq,rnn
import numpy as np


class Seq2SeqModel():
    def __init__(self, rnn_size, num_layers, embedding_size, learning_rate, learning_rate_decay_factor, word_to_idx, mode, use_attention,
                 beam_search, beam_size, max_gradient_norm=5.0):
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.word_to_idx = word_to_idx
        self.vocab_size = len(self.word_to_idx)
        self.mode = mode
        self.use_attention = use_attention
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.max_gradient_norm = max_gradient_norm
        #执行模型构建部分的代码
        self.build_model()

    def basic_rnn_cell(self):
        return tf.contrib.rnn.LSTMCell(self.rnn_size)


    def _create_decoder_cell(self):
        def single_rnn_cell():
            single_cell = rnn.LSTMCell(self.rnn_size*2)
            cell = rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder)
            return cell
        cell = rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell

    def build_model(self):
        print('building model... ...')
        #=================================1, 定义模型的placeholder
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')

        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')

        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        self.mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_sequence_length, dtype=tf.float32, name='masks')

        #=================================2, 定义模型的encoder部分
        with tf.variable_scope('encoder'):
            wordembedding = np.load('../data/wordEmbedding300.npy').astype('float32')
            embedding = tf.get_variable('embedding', dtype=tf.float32,
                                        initializer=wordembedding,
                                        trainable=True)
            encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, self.encoder_inputs)

            with tf.name_scope('fw_rnn'):
                encoder_fw_cell = tf.contrib.rnn.MultiRNNCell(
                    [self.basic_rnn_cell() for _ in range(self.num_layers)])
                encoder_fw_cell = tf.contrib.rnn.DropoutWrapper(encoder_fw_cell,
                                                                output_keep_prob=self.keep_prob_placeholder)

            with tf.name_scope('bw_rnn'):
                encoder_bw_cell = tf.contrib.rnn.MultiRNNCell(
                    [self.basic_rnn_cell() for _ in range(self.num_layers)])
                encoder_bw_cell = tf.contrib.rnn.DropoutWrapper(encoder_bw_cell,
                                                                output_keep_prob=self.keep_prob_placeholder)

            (encoder_fw_outputs, encoder_bw_outpus), \
            (encoder_fw_final_status, encoder_bw_final_status) = tf.nn.bidirectional_dynamic_rnn(encoder_fw_cell,
                                                                                                 encoder_bw_cell,
                                                                                                 inputs=encoder_inputs_embedded,
                                                                                                 sequence_length=self.encoder_inputs_length,
                                                                                                 dtype=tf.float32)



            encoder_outputs = tf.concat([encoder_fw_outputs, encoder_bw_outpus], axis=2)

            c0 = tf.concat(
                [encoder_fw_final_status[0].c, encoder_bw_final_status[0].c], axis=1)
            h0 = tf.concat(
                [encoder_fw_final_status[0].h, encoder_bw_final_status[0].h], axis=1)

            c1 = tf.concat(
                [encoder_fw_final_status[-1].c, encoder_bw_final_status[-1].c], axis=1)
            h1 = tf.concat(
                [encoder_fw_final_status[-1].h, encoder_bw_final_status[-1].h], axis=1)

            encoder_state0 = rnn.LSTMStateTuple(c=c0,
                                               h=h0)
            encoder_state1 = rnn.LSTMStateTuple(c=c1,
                                                h=h1)

            encoder_state = tuple((encoder_state0, encoder_state1))
            # print("encode_state====>",encoder_state)
            print('encoder_outputs',encoder_outputs)
            print('encoder_state',encoder_state)
        # =================================3, 定义模型的decoder部分
        with tf.variable_scope('decoder'):
            encoder_inputs_length = self.encoder_inputs_length
            if self.beam_search:
                # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
                print("use beamsearch decoding..")
                encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_size)
                encoder_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_size), encoder_state)
                encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.encoder_inputs_length, multiplier=self.beam_size)

            #定义要使用的attention机制。
            attention_mechanism = seq2seq.BahdanauAttention(num_units=self.rnn_size, memory=encoder_outputs,
                                                                     memory_sequence_length=encoder_inputs_length)

            decoder_cell = self._create_decoder_cell()
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism,
                                                               attention_layer_size=self.rnn_size, name='Attention_Wrapper')
            batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size
            decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_state)
            output_layer = tf.layers.Dense(self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            if self.mode == 'train':
                ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
                decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word_to_idx['_GO']), ending], 1)
                decoder_inputs_embedded = tf.nn.embedding_lookup(embedding, decoder_input)
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                                    sequence_length=self.decoder_targets_length,
                                                                    time_major=False, name='training_helper')
                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=training_helper,
                                                                   initial_state=decoder_initial_state, output_layer=output_layer)

                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                          impute_finished=True,
                                                                    maximum_iterations=self.max_target_sequence_length)

                self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
                self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train')
                self.loss = seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                             targets=self.decoder_targets, weights=self.mask)

                # Training summary for the current batch_loss
                tf.summary.scalar('loss', self.loss)
                self.summary_op = tf.summary.merge_all()

                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                #获取待训练的参数
                trainable_params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, trainable_params)
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

                self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

            elif self.mode == 'decode':
                start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word_to_idx['_GO']
                end_token = self.word_to_idx['_EOS']
                if self.beam_search:
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell, embedding=embedding,
                                                                             start_tokens=start_tokens, end_token=end_token,
                                                                             initial_state=decoder_initial_state,
                                                                             beam_width=self.beam_size,
                                                                             output_layer=output_layer)
                else:
                    # 这是用于inference阶段的helper，将output输出后的logits使用argmax获得id再经过embedding layer来获取下一时刻的输入。
                    decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                                               start_tokens=start_tokens, end_token=end_token)
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=decoding_helper,
                                                                        initial_state=decoder_initial_state,
                                                                        output_layer=output_layer)
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                maximum_iterations=10)
                if self.beam_search:
                    self.decoder_predict_decode = decoder_outputs.predicted_ids

                else:
                    self.decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)
        # =================================4, 保存模型
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=15,
                                    pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def train(self, sess, batch):
        #对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.decoder_targets: batch.decoder_targets,
                      self.decoder_targets_length: batch.decoder_targets_length,
                      self.keep_prob_placeholder: 0.5,
                      self.batch_size: len(batch.encoder_inputs)}
        _,learning_rate, loss, summary = sess.run([self.train_op, self.learning_rate,self.loss, self.summary_op], feed_dict=feed_dict)
        return learning_rate,loss, summary

    def eval(self, sess, batch):
        # 对于eval阶段，不需要反向传播，所以只执行self.loss, self.summary_op两个op，并传入相应的数据
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.decoder_targets: batch.decoder_targets,
                      self.decoder_targets_length: batch.decoder_targets_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(batch.encoder_inputs)}
        learning_rate, loss, summary = sess.run([self.learning_rate,self.loss, self.summary_op], feed_dict=feed_dict)
        return learning_rate, loss, summary

    def infer(self, sess, batch):
        #infer阶段只需要运行最后的结果，不需要计算loss，所以feed_dict只需要传入encoder_input相应的数据即可
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(batch.encoder_inputs)}
        predict = sess.run([self.decoder_predict_decode], feed_dict=feed_dict)
        return predict