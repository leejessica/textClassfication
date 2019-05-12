# -*- coding: utf-8 -*-

import tensorflow as tf
import logging
from .layer import conv, batch_norm, relu, max_pool, linear
import shutil
import os
from data_provider import load_data
from tqdm import tqdm

logging.basicconfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextCNN:
    def __init__(self, config, batch, optimizer='Adam'):
        self.config = config
        self.filter_sizes = config.filter_sizes
        self.num_classes = config.num_classes
        self.batch_size = config.batch_size
        self.sequence_length = config.max_sequence_length
        self.vocab_size = config.vocab_size
        self.learning_rate = tf.Variable(config.learning_rate, trainable=False,
                                         name="learning_rate")  # ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * config.decay_rate_big)
        self.num_filters = config.num_filters
        self.initializer = config.initializer
        self.linear_hidden_size = config.linear_hidden_size
        self.pool_size = config.pool_size
        self.global_step = tf.get_variable('global_step', dtype=tf.int32, initializer=tf.constant_initializer(0),
                                           trainable=False)
        if config.use_word:
            self.embed_size = config.word_dim
            self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(config.word_mat, dtype=tf.float32),
                                            trainable=False)
        if config.multi_label_flag:
            if config.is_demo:
                self.x = tf.placeholder("float", shape=[None, None, self.embed_size], name="x")
                self.y = tf.placeholder("float", shape=[None, self.num_classes], name="y")
            else:
                self.x, self.y = batch.get_next()
            self.loss_val = self.loss_multilabel()
        else:
            if config.is_demo:
                self.x = tf.placeholder("float", shape=[None, None, self.embed_size], name="x")
                self.y = tf.placeholder("float", shape=[None], name="y")
            else:
                self.x, self.y = batch.get_next()
            self.loss_val = self.loss()

        self.is_training_flag = tf.placeholder(tf.bool, name="is_training_flag")

        self.logits = self.forward(self.x)
        self.loss = self.loss_val(logits=self.logits, label=self.y)

        if self.is_training_flag:
            opt_list = ['Adagrad', 'Adam', 'Ftrl', 'Momentum', 'RMSProp', 'SGD']
            if optimizer not in opt_list:
                raise ValueError('optimizer name must in:', ', '.join(str(p) for p in opt_list))
            self.learning_rate = tf.train.exponential_decay(config.learning_rate, self.global_step, config.decay_steps,
                                                            config.decay_rate, staircase=True)
            self.train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                            learning_rate=self.learning_rate, optimizer="Adam",
                                                            clip_gradients=config.clip_gradients)

    def encoder_block(self, inputs, filter_size, pool_size, reuse=None):
        """
        block: CONV->BN->RELU->CONV->BN->RELU->MaxPool

        """

        with tf.variable_scope("Encoder_Block_%d" % filter_size, reuse=reuse):
            conv1 = conv(inputs, self.num_filters, bias=True, kernel_size=filter_size, name="conv_1", reuse=reuse)
            bn1 = batch_norm(conv1, name="BN_1", train=self.is_training_flag, reuse=reuse)
            relu1 = relu(bn1, name="relu_1", reuse=reuse)
            conv2 = conv(relu1, self.num_filters, bias=True, kernel_size=filter_size, name="conv_2", reuse=reuse)
            bn2 = batch_norm(conv2, name="BN_2", train=self.is_training_flag, reuse=reuse)
            relu2 = relu(bn2, name="relu_2", reuse=reuse)
            outputs = max_pool(relu2, pool_size=pool_size, reuse=reuse)
            return outputs

    def forward(self, reuse=None):

        with tf.variable_scope("Input_Embedding_Layer", reuse=reuse):
            if self.config.use_word:
                word_emb = tf.reshape(tf.nn.embedding_lookup(self.word_mat, self.x),
                                      [self.batch_size, self.config.max_sequence_length, self.embed_size])

        with tf.variable_scope("TextCNN", reuse=reuse):
            block_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                block_outputs[i] = self.encoder_block(word_emb, filter_size, self.pool_size, reuse=reuse)
            concat = tf.concat(block_outputs, axis=1, name='concat')
            flatten = tf.layers.flatten(concat)
            linear1 = linear(flatten, units=self.linear_hidden_size, name='linear_1', reuse=reuse)
            bn1 = batch_norm(linear1, name='linear_BN', train=train, reuse=reuse)
            relu1 = relu(bn1, name='linear_relu', reuse=reuse)
            logits = linear(relu1, units=self.linear_hidden_size, name='linear_2', reuse=reuse)
            return logits

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

    epoches = config.epoches
    save_path = os.path.join(config.textCNN_path, "model.ckpt")

    graph = tf.Graph()

    with graph.as_default() as g:
        train_dataset = load_data(config.train_data_path)
        dev_dataset = load_data(config.dev_data_path)
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

        model = TextCNN(config, iterator, optimizer)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

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



            for _ in tqdm(range(global_step, config.num_steps + 1)):
                global_step = sess.run(model.global_step) + 1
                loss, train_op = sess.run([model.loss_val, model.train_op], feed_dict={
                    handle: train_handle, model.dropout: config.dropout})
                if global_step % config.period == 0:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(
                        tag="model/loss", simple_value=loss), ])
                    writer.add_summary(loss_sum, global_step)
                if global_step % config.checkpoint == 0:
                    _, summ = evaluate_batch(
                        model, config.val_num_batches, train_eval_file, sess, "train", handle, train_handle)
                    for s in summ:
                        writer.add_summary(s, global_step)

                    metrics, summ = evaluate_batch(
                        model, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)

                    dev_f1 = metrics["f1"]
                    dev_em = metrics["exact_match"]
                    if dev_f1 < best_f1 and dev_em < best_em:
                        patience += 1
                        if patience > config.early_stop:
                            break
                    else:
                        patience = 0
                        best_em = max(best_em, dev_em)
                        best_f1 = max(best_f1, dev_f1)

                    for s in summ:
                        writer.add_summary(s, global_step)
                    writer.flush()
                    filename = os.path.join(
                        config.save_dir, "model_{}.ckpt".format(global_step))
                    saver.save(sess, filename)
