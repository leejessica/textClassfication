import tensorflow as tf
from glob import glob
import os


def build_text_data(input_file, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    with open(input_file, 'r') as fh:
        lines = fh.readlines()
    for line in lines:
        sample = line.split('_!_')
        text = sample[3].encode()
        label = int(sample[1])
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "text": tf.train.Feature(bytes_list=tf.train.BytesList(value=[text])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                }
            )
        )
        writer.write(record=example.SerializeToString())
    writer.close()


def batch_build_text_data(input_dir, output_dir):
    for input_file in glob(os.path.abspath(input_dir) + "/*.txt"):
        output_file = os.path.abspath(output_dir).join(os.path.basename(input_file).split('.')[0])
        build_text_data(input_file, output_file)
