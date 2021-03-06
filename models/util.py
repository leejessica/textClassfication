import tqdm
import tensorflow as tf
import numpy as np


def evaluate(labels, prediction):
    _, precision = tf.metrics.accuracy(labels=labels, predictions=prediction)
    _, recall = tf.metrics.recall(labels=labels, predictions=prediction)
    f1 = (precision + recall) / 2 * precision * recall
    return {'f1': f1, 'precision': precision, 'recall': recall}


def evaluate_batch(model, num_batches, sess, data_type, handle, str_handle):
    lables=[]
    predictions=[]
    losses = []
    for _ in tqdm(range(1, num_batches + 1)):
        loss, label, prediction= sess.run(
            [model.loss_val, model.y, model.ligits], feed_dict={handle: str_handle, model.is_training_flag: False})
        losses.append(loss)
        lables.append(label)
        predictions.append(prediction)
    loss = np.mean(losses)
    metrics = evaluate(lables, predictions)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    pre_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/precision".format(data_type), simple_value=metrics["precision"]), ])
    recall_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/recall".format(data_type), simple_value=metrics["recall"]), ])
    return metrics, [loss_sum, pre_sum, recall_sum, f1_sum]


def get_record_parser(config):
    def parse(example):
        features = tf.parse_single_example(example,
                                           features={
                                               "text_idxs": tf.FixedLenFeature([], tf.string),
                                               "text_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "label": tf.FixedLenFeature([], tf.string)
                                           })
        text_idxs = tf.reshape(tf.decode_raw(
            features["text_idxs"], tf.int32), [config.max_sequence_length])
        text_char_idxs = tf.reshape(tf.decode_raw(
            features["text_char_idxs"], tf.int32), [config.max_sequence_length, config.char_limit])
        label = features["label"]
        return text_idxs, text_char_idxs, label

    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
    if config.is_bucket:
        buckets = [tf.constant(num) for num in range(*config.bucket_range)]

        def key_func(text_idxs, label):
            c_len = tf.reduce_sum(
                tf.cast(tf.cast(text_idxs, tf.bool), tf.int32))
            t = tf.clip_by_value(buckets, 0, c_len)
            return tf.argmax(t)

        def reduce_func(key, elements):
            return elements.batch(config.batch_size)

        dataset = dataset.apply(tf.contrib.data.group_by_window(
            key_func, reduce_func, window_size=config.batch_size)).shuffle(len(buckets) * 25)
    else:
        dataset = dataset.batch(config.batch_size)
    return dataset
