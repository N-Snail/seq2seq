# -*- coding: utf-8 -*-

import tensorflow as tf
from data_helper import load_dict,loadDataset, getBatches, sentence2enco
from seq2seq_model import Seq2SeqModel
from tqdm import tqdm
import math
import os

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
# data_path = '../data/train_data_idx.pkl'
data_path = '../data/train_new_data_idx.pkl'
trainingSamples = loadDataset(data_path)

# test_path = '../data/test_data_idx.pkl'
test_path = '../data/test_new_data_idx26.pkl'
testingSamples = loadDataset(test_path)


model = Seq2SeqModel(FLAGS.rnn_size, FLAGS.num_layers, FLAGS.embedding_size, FLAGS.learning_rate,FLAGS.learning_rate_decay_factor, word2idx,
                         mode='train', use_attention=True, beam_search=False, beam_size=5, max_gradient_norm=5.0)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print('Reloading model parameters..')
    model.saver.restore(sess, ckpt.model_checkpoint_path)
else:
    print('Created new model parameters..')
    sess.run(tf.global_variables_initializer())

current_step = 0
loss = 0.0
previous_losses = []

summary_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)
for e in range(FLAGS.numEpochs):
    print("----- Epoch {}/{} -----".format(e + 1, FLAGS.numEpochs))
    batches = getBatches(trainingSamples, FLAGS.batch_size)
    for nextBatch in tqdm(batches, desc="Training"):
        learning_rate, step_loss, summary = model.train(sess, nextBatch)
        current_step += 1
        loss += step_loss / 100
        if current_step % FLAGS.steps_per_checkpoint == 0:
            perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
            tqdm.write("----- Step %d -- Learning_rate %f -- Loss %.4f -- Perplexity %.4f" % (
            current_step, learning_rate, loss, perplexity))
            summary_writer.add_summary(summary, current_step)
            checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
            model.saver.save(sess, checkpoint_path)
            if len(previous_losses) > 10 and loss > max(previous_losses[-11:]):
                sess.run(model.learning_rate_decay_op)
            previous_losses.append(loss)
            loss = 0.0

    checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
    model.saver.save(sess, checkpoint_path, global_step=current_step)

    test_batches = getBatches(testingSamples, FLAGS.batch_size)
    test_losses = []
    for nextBatch in tqdm(test_batches, desc="Testing"):
        test_learning_rate, step_loss, summary = model.eval(sess, nextBatch)
        test_losses.append(step_loss)
    testloss = float(sum(test_losses)) / float(len(test_losses))
    test_perplexity = math.exp(float(testloss)) if loss < 300 else float('inf')
    tqdm.write("test -- Loss %.4f -- Perplexity %.4f" % (testloss, test_perplexity))

