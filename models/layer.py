import tensorflow as tf
import numpy as np

initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                     mode='FAN_AVG',
                                                                     uniform=True,
                                                                     dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                          mode='FAN_IN',
                                                                          uniform=False,
                                                                          dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)


def conv(inputs, output_size, bias=None, activation=None, kernel_size=1, name="conv", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1, kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, 1, output_size]
            strides = [1, 1, 1, 1]
        else:
            filter_shape = [kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_",
                                  filter_shape,
                                  dtype=tf.float32,
                                  regularizer=regularizer,
                                  initializer=initializer_relu() if activation is not None else initializer())
        outputs = conv_func(inputs, kernel_, strides, "VALID")
        if bias:
            outputs += tf.get_variable("bias_",
                                       bias_shape,
                                       regularizer=regularizer,
                                       initializer=tf.zeros_initializer())
        return outputs


def max_pool(inputs, pool_size=1, name="max_pool", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            pool_shape = [1, pool_size]
            strides = [1, 1]
        else:
            pool_shape = pool_size
            strides = 1
        pool_func = tf.layers.max_pooling1d if len(shapes) == 3 else tf.layers.max_pooling2d
        outputs = pool_func(inputs, pool_shape, strides, "VALID")
        return outputs


def batch_norm(inputs, name="BN", train=True, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        outputs = tf.layers.batch_normalization(inputs, training=train, reuse=reuse)
        return outputs


def relu(inputs, name="relu", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        return tf.nn.relu(inputs)


def linear(inputs, units, name="linear", activation=None, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        outputs = tf.layers.dense(inputs,
                                  units=units,
                                  activation=activation,
                                  kernel_initializer=initializer_relu() if activation is not None else initializer(),
                                  kernel_regularizer=regularizer(),
                                  name=name,
                                  reuse=reuse
                                  )
        return outputs
