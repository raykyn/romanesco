#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.ops.functional_ops import map_fn

from romanesco.const import *


def define_computation_graph(vocab_size: int, batch_size: int):

    # Placeholders for input and output
    inputs = tf.placeholder(tf.int32, shape=(batch_size, NUM_STEPS), name='x')  # (time, batch)
    targets = tf.placeholder(tf.int32, shape=(batch_size, NUM_STEPS), name='y') # (time, batch)
    
    init_state = tf.placeholder(tf.float32, [NUM_LAYERS, 2, batch_size, STATE_SIZE])
    
    
    state_per_layer_list = tf.unstack(init_state, axis=0)
    rnn_tuple_state = tuple(
        [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
         for idx in range(NUM_LAYERS)]
    )

    with tf.name_scope('Embedding'):
        embedding = tf.get_variable('word_embedding', [vocab_size, STATE_SIZE])
        input_embeddings = tf.nn.embedding_lookup(embedding, inputs)

    with tf.name_scope('RNN'):
        cell = tf.nn.rnn_cell.LSTMCell(STATE_SIZE, state_is_tuple=True)
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * NUM_LAYERS, state_is_tuple=True)
        rnn_outputs, current_state = tf.nn.dynamic_rnn(cell, input_embeddings, initial_state=rnn_tuple_state)
        #rnn_outputs = tf.reshape(rnn_outputs, [-1, STATE_SIZE])

    with tf.name_scope('Final_Projection'):
        w = tf.get_variable('w', shape=(STATE_SIZE, vocab_size))
        b = tf.get_variable('b', vocab_size)
        final_projection = lambda x: tf.matmul(x, w) + b
        logits = map_fn(final_projection, rnn_outputs)
        #targets = tf.reshape(targets, [-1])

    with tf.name_scope('Cost'):
        # weighted average cross-entropy (log-perplexity) per symbol
        loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                targets=targets,
                                                weights=tf.ones([batch_size, NUM_STEPS]),
                                                average_across_timesteps=True,
                                                average_across_batch=True)

    with tf.name_scope('Optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    # Logging of cost scalar (@tensorboard)
    tf.summary.scalar('loss', loss)
    summary = tf.summary.merge_all()

    return inputs, targets, loss, train_step, logits, summary, current_state, init_state
