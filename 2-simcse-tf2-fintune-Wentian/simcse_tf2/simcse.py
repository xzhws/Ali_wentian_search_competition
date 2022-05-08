# -*- coding:utf-8 -*-
"""
Author:
    jifei, jifei@outlook.com
"""
import tensorflow as tf
import os
import json
import keras.backend as K

# os.environ["TF_KERAS"] = '1'
from bert4keras.models import build_transformer_model


def simcse(config_path,
           checkpoint_path,
           model='bert',
           pooling='first-last-avg',
           dropout_rate=0.1,
           output_units=None,
           output_activation=None,
           ):
    """Build SimCSE model

    :param config_path:string
    :param checkpoint_path:string
    :param model:string, model name
    :param pooling:string, in ['first-last-avg', 'last-avg', 'cls', 'pooler']
    :param dropout_rate:float
    :param output_units:int
    :param output_activation:string
    :return: A Keras model instance.
    """
    assert pooling in ['first-last-avg', 'last-avg', 'cls', 'pooler']
    with open(config_path, 'r') as load_f:
        num_hidden_layers = json.load(load_f)['num_hidden_layers']

    # if pooling == 'pooler':
    #     bert = build_transformer_model(
    #         config_path,
    #         checkpoint_path,
    #         model=model,
    #         with_pool='linear',
    #         dropout_rate=dropout_rate
    #     )
    # else:
    # bert = build_transformer_model(config_path, checkpoint_path, model='bert', dropout_rate=dropout_rate)

    bert = build_transformer_model(
        config_path,
        checkpoint_path,
        model='roformer',
        application='unilm',
        with_pool='linear',
        with_mlm='linear',
        dropout_rate=dropout_rate,
        ignore_invalid_weights=True,
        return_keras_model=False,
    )
    # print(bert.summary())

    # print('bert output:{}'.format(bert.output))
    print('num_hidden_layers:{}'.format(num_hidden_layers))

    last_layer_output = bert.model.get_layer('Transformer-%d-FeedForward-Norm' % (num_hidden_layers - 1)).output
    print('last_layer_output:{}'.format(last_layer_output))

    if pooling == 'first-last-avg':
        outputs = [
            tf.keras.layers.GlobalAveragePooling1D()(bert.model.get_layer('Transformer-%d-FeedForward-Norm' % 0).output),
            tf.keras.layers.GlobalAveragePooling1D()(last_layer_output)
        ]
        output = tf.keras.layers.Average()(outputs)
    elif pooling == 'last-avg':
        output = tf.keras.layers.GlobalAveragePooling1D()(last_layer_output)
    elif pooling == 'cls':
        output = tf.keras.layers.Lambda(lambda x: x[:, 0])(last_layer_output)
    else:
        output = bert.output

    if output_units and output_activation:
        output = tf.keras.layers.Dense(output_units, activation=output_activation)(output)
    # normlize:
    output = K.l2_normalize(output, axis=-1)
    # print('output shape:{}'.format(K.int_shape(output)))
    model = tf.keras.Model(bert.inputs, output)
    return model
