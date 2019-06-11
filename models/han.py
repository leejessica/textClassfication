# -*- coding: utf-8 -*-

import tensorflow as tf
import logging
from .layer import bi_gru, attention, linear
import shutil
import os
from tqdm import tqdm
from .util import evaluate_batch, evaluate, get_batch_dataset, get_record_parser
import numpy as np

logging.basicconfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HierarchicalAttention:
    def __init__(self, config, batch=None, optimizer='Adam', graph=None):
        self.graph = graph if graph is not None else tf.Graph()
        with self.graph.as_default():
            self.config = config
            self.bilstm_hidden_size = config.bilstm_hidden_size
            self.singlelstm_hidden_size = config.singlelstm_hidden_size
            self.num_classes = config.num_classes
            self.batch_size = config.batch_size
            self.sequence_length = config.max_sequence_length
            self.max_sentence_num = config.han_max_sentece_num
            self.vocab_size = config.vocab_size
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False,
                                             name="learning_rate")  # ADD learning_rate
            self.dropout_keep_prob = config.drop_keep_pro
            self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * config.decay_rate_big)
            self.initializer = config.initializer
            self.linear_hidden_size = config.linear_hidden_size
            self.pool_size = config.pool_size
            self.global_step = tf.get_variable('global_step', dtype=tf.int32, initializer=tf.constant_initializer(0),
                                               trainable=False)
            self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(config.word_mat, dtype=tf.float32),
                                            trainable=False)

            if config.multi_label_flag:
                if config.is_demo:
                    self.x_word = tf.placeholder("float", shape=[None, self.max_sentence_num, self.sequence_length],
                                                 name="x_word")
                    self.y = tf.placeholder("float", shape=[None, self.num_classes], name="y")
                else:
                    self.x_word, _, self.y = batch.get_next()
                    self.loss_val = self.loss_multilabel()
            else:
                if config.is_demo:
                    self.x_word = tf.placeholder("float", shape=[None, self.max_sentence_num, self.sequence_length],
                                                 name="x_word")
                    self.y = tf.placeholder("float", shape=[None], name="y")
                else:
                    self.x_word, _, self.y = batch.get_next()
                    self.loss_val = self.loss()

            self.is_training_flag = tf.placeholder(tf.bool, name="is_training_flag")

            self.forward()
            self.loss = self.loss_val(logits=self.logits, label=self.y)

            if self.is_training_flag:
                opt_list = ['Adagrad', 'Adam', 'Ftrl', 'Momentum', 'RMSProp', 'SGD']
                if optimizer not in opt_list:
                    raise ValueError('optimizer name must in:', ', '.join(str(p) for p in opt_list))
                self.learning_rate = tf.train.exponential_decay(config.learning_rate, self.global_step,
                                                                config.decay_steps,
                                                                config.decay_rate, staircase=True)
                self.train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                                learning_rate=self.learning_rate, optimizer="Adam",
                                                                clip_gradients=config.clip_gradients)

    def forward(self, reuse=None):
        with tf.variable_scope("Input_Embedding_Layer"):
            word_emb = tf.reshape(tf.nn.embedding_lookup(self.word_mat, self.x_word),
                                  [self.batch_size, self.max_sentence_num, self.sequence_length, self.config.word_dim])
            word_emb = [tf.squeeze(x) for x in tf.split(word_emb, self.max_sentence_num,
                                                        axis=1)]  # a list.length is max_sentence_num, each element is:[None,sequence_length,embed_size]

        word_attention_list = []
        for i in range(self.max_sentence_num):
            sentence = word_emb[i]  # [batch_size, sequence_length, embed_size]
            reuse_flag = True if i > 0 else False
            word_encoded, _ = bi_gru(sentence, self.config.word_dim, 'word_level', self.dropout_keep_prob,
                                  reuse=reuse_flag)
            word_attention = attention(word_encoded, 'word_attention', reuse=reuse_flag)
            word_attention_list.append(word_attention)



        if self.config.decay is not None:
            self.var_ema = tf.train.ExponentialMovingAverage(self.config.decay)
            ema_op = self.var_ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.loss = tf.identity(self.loss)

                self.assign_vars = []
                for var in tf.global_variables():
                    v = self.var_ema.average(var)
                    if v:
                        self.assign_vars.append(tf.assign(var, v))


def loss_multilabel(self, labels, logits, l2_lambda=0.0001):
    with tf.name_scope("loss"):
        # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
        # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
        # input_y:shape=(?, 1999); logits:shape=(?, 1999)
        # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                         logits=logits);
        logger.info("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(batch_size, num_classes).
        losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
        loss = tf.reduce_mean(losses)  # shape=().   average loss in the batch
        l2_losses = tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss = loss + l2_losses
    return loss


def loss(self, labels, logits, l2_lambda=0.0001):
    with tf.name_scope("loss"):
        # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
        # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                logits=logits)
        loss = tf.reduce_mean(losses)
        l2_losses = tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss = loss + l2_losses
    return loss


def train(config, optimizer='Adam', restore=False):
    """
    :param optimizer: name of optimizer, full list in OPTIMIZER_CLS_NAMES constant
    :param restore: Flag if previous model should be restored
    :return:
    """

    if not restore:
        logger.info("Removing '{:}'".format(config.textCNN_path))
        shutil.rmtree(config.model_path, ignore_errors=True)

    if not os.path.exists(config.model_path):
        logger.info("Allocating '{:}'".format(config.textCNN_path))

    graph = tf.Graph()

    parser = get_record_parser(config)

    with graph.as_default() as g:
        train_dataset = get_batch_dataset(config.train_record_file, parser, config)
        dev_dataset = get_batch_dataset(config.dev_record_file, parser, config)
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types,
                                                       train_dataset.output_shapes)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

        model = TextRNN(config=config, batch=iterator, optimizer=optimizer, graph=g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        patience = 0
        best_f1 = 0.

        with tf.Session(config=sess_config) as sess:
            writer = tf.summary.FileWriter(config.textCNN_log)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            train_handle = sess.run(train_iterator.string_handle())
            dev_handle = sess.run(dev_iterator.string_handle())
            if os.path.exists(os.path.join(config.textCNN_path, "checkpoint")):
                saver.restore(sess, tf.train.latest_checkpoint(config.textCNN_path))
            global_step = max(sess.run(model.global_step), 1)
            logger.info("global_step = %s".format(global_step))
            # todo fix train_dataset.output_shapes[0] bug
            num_steps = config.epoches * (tf.cast(train_dataset.output_shapes[0] / config.batch_size, tf.int32) + 1)
            for iter in tqdm(range(0, num_steps)):
                loss, train_op = sess.run([model.loss_val, model.train_op], feed_dict=
                {handle: train_handle, model.is_training_flag: True})
                if iter % config.period == 0:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss), ])
                    writer.add_summary(loss_sum, iter)
                if iter % config.checkpoint == 0:
                    _, summ = evaluate_batch(model, config.val_num_batches, sess, "train", handle,
                                             train_handle)
                    for s in summ:
                        writer.add_summary(s, iter)
                    metrics, summ = evaluate_batch(model, config.dev_num_bathes, sess, "dev", handle,
                                                   dev_handle)
                    dev_f1 = metrics["f1"]
                    if dev_f1 < best_f1:
                        patience += 1
                        if patience > config.early_stop:
                            break
                    else:
                        patience = 0
                        best_f1 = max(best_f1, dev_f1)
                    for s in summ:
                        writer.add_summary(s, iter)
                    writer.flush()
                    filename = os.path.join(config.textCNN_path, "model_{}.ckpt".format(iter))
                    saver.save(sess, filename)


def test(config):
    graph = tf.Graph()
    logger.info("Loading model ...")
    with graph.as_default() as g:
        test_batch = get_batch_dataset(config.test_record_file, get_record_parser(
            config), config).make_one_shot_iterator()

        model = TextRNN(config=config, batch=test_batch, graph=g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(config.textCNN_path))
            if config.decay < 1.0:
                sess.run(model.assign_vars)

            losses = []

            for _ in tqdm(range(config.test_steps)):
                loss, y, logits = sess.run([model.loss_val, model.y, model.logits],
                                           feed_dict={model.is_training_flag: False})
                losses.append(loss)
            loss = np.mean(losses)
            metrics = evaluate(labels=y, prediction=logits)

            logger.info("precision: {}, recall:{}, F1: {}, average_loss:{}".format(
                metrics['precision'], metrics['recall'], metrics['f1'], loss))
