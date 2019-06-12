import tensorflow as tf

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


def bi_lstm(inputs, hidden_size, name="Bi-lstm", dropout_keep_prob=None, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
        if dropout_keep_prob is not None:
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=dropout_keep_prob)
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=dropout_keep_prob)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs)
        outputs=tf.concat(outputs, axis=-1)
    return outputs, states


def lstm(inputs, hidden_size, name="Single-lstm", dropout_keep_prob=None, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
        if dropout_keep_prob is not None:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=dropout_keep_prob)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputs)
    return outputs, states


def bi_gru(inputs, hidden_size, encoder_level, dropout_keep_prob=None, reuse=None):
    """

    :param inputs: [batch_size, seq_length, embed_size]
    :param hidden_size: number of GRU cell
    :param name: scope name
    :param dropout_keep_prob:
    :param reuse:  (optional) Python boolean describing whether to reuse variables in an existing scope. If not True, and the existing scope already has the given variables, an error is raised.
    :return:
    """
    with tf.variable_scope("encoder_" + str(encoder_level), reuse=reuse):
        gru_fw_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
        gru_bw_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
        if dropout_keep_prob is not None:
            gru_fw_cell = tf.nn.rnn_cell.DropoutWrapper(gru_fw_cell, output_keep_prob=dropout_keep_prob)
            gru_bw_cell = tf.nn.rnn_cell.DropoutWrapper(gru_bw_cell, output_keep_prob=dropout_keep_prob)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(gru_fw_cell, gru_bw_cell, inputs)
            outputs=tf.concat(outputs, axis=-1)
    return outputs, states


def attention(self, input_sequences, attention_level, reuse=False):
    """
    :param input_sequence: [batch_size,seq_length,num_units]
    :param attention_level: word or sentence level
    :return: [batch_size,hidden_size]
    """
    num_units = input_sequences.get_shape().as_list()[-1]  # get last dimension
    with tf.variable_scope("attention_" + str(attention_level), reuse=reuse):
        v_attention = tf.get_variable("u_attention" + attention_level, shape=[num_units], initializer=self.initializer)
        # 1.one-layer MLP
        u = tf.layers.dense(input_sequences, num_units, activation=tf.nn.tanh,
                            use_bias=True)  # [batch_size,seq_legnth,num_units].no-linear
        # 2.compute weight by compute simility of u and attention vector v
        score = tf.multiply(u, v_attention)  # [batch_size,seq_length,num_units]
        weight = tf.reduce_sum(score, axis=-1, keepdims=True)  # [batch_size,seq_length,1]
        # 3.weight sum
        attention_representation = tf.reduce_sum(tf.multiply(u, weight), axis=1)  # [batch_size,num_units]
    return attention_representation


