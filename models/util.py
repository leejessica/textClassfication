import tqdm
import tensorflow as tf
import numpy as np



def evaluate(eval_file, answer_dict):
    f1 = precision = recall = 0
    for key, value in answer_dict.items():
        ground_truths = eval_file[key]["label"]
        prediction = value
        _, precision = tf.metrics.accuracy(labels=ground_truths, predictions=prediction)
        _, recall = tf.metrics.recall(labels=ground_truths, predictions=prediction)
        f1 = (precision + recall) / 2 * precision * recall
    return {'f1': f1, 'precision': precision, 'recall': recall}


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    answer_dict = {}
    losses = []
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, y= sess.run(
            [model.qa_id, model.loss_val, model.y], feed_dict={handle: str_handle, model.is_training_flag: False})
        # Todo answer_dict_=
        answer_dict_={}
        answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    pre_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/precision".format(data_type), simple_value=metrics["precision"]), ])
    recall_sum=tf.Summary(value=[tf.Summary.Value(
        tag="{}/recall".format(data_type), simple_value=metrics["recall"]),])
    return metrics, [loss_sum, pre_sum, recall_sum, f1_sum]
