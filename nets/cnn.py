# -*- coding: utf-8 -*-
from nets.common import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import layers as tflayers
from tensorflow.contrib import learn as tflearn_old
import tflearn


class RCNN_BaseLine(BaseModel):
    def __init__(self, depth=6, init_channels=16, input_frames=20,
                 input_height=128, input_width=128, input_channels=1, hiddenDim=256):
        self.depth = depth
        self.input_frames = input_frames
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.init_channels = init_channels
        self.vectorSize = input_height * input_width * input_frames
        self.hiddenDim = hiddenDim

    def resnetLayer(self, inputs, training):
        self.resnet_size = 18,
        self.bottleneck = True,
        self.num_filters = 64,
        self.kernel_size = 7,
        self.conv_stride = 2,
        self.first_pool_size = 3,
        self.first_pool_stride = 2,
        self.second_pool_size = 7,
        self.second_pool_stride = 1,
        self.block_sizes = _get_block_sizes(resnet_size),
        self.block_strides = [1, 2, 2, 2],
        self.final_size = final_size,

        self.data_format = data_format

        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
            strides=self.conv_stride, data_format=self.data_format)
        inputs = tf.identity(inputs, 'initial_conv')

        inputs = tf.layers.max_pooling2d(
            inputs=inputs, pool_size=self.first_pool_size,
            strides=self.first_pool_stride, padding='SAME',
            data_format=self.data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')

        for i, num_blocks in enumerate(self.block_sizes):
            num_filters = self.num_filters * (2 ** i)
            inputs = block_layer(
                inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
                block_fn=self.block_fn, blocks=num_blocks,
                strides=self.block_strides[i], training=training,
                name='block_layer{}'.format(i + 1), data_format=self.data_format)

        inputs = batch_norm(inputs, training, self.data_format)
        inputs = tf.nn.relu(inputs)
        inputs = tf.layers.average_pooling2d(
            inputs=inputs, pool_size=self.second_pool_size,
            strides=self.second_pool_stride, padding='VALID',
            data_format=self.data_format)
        inputs = tf.identity(inputs, 'final_avg_pool')

        return inputs

    def vggLayer(self, input, name, reuse):

        # input image = 128 128
        # vgg-11
        with tf.variable_scope('vggLayer_' + name, reuse=reuse):
            network = tflearn.conv_2d(input, 16, 3, activation='relu', name='conv1_1')
            network = tflearn.conv_2d(network, 16, 3, activation='relu', name='conv1_2')
            network = tflearn.max_pool_2d(network, 2, strides=2)
            network = tflearn.batch_normalization(network)
            # 64 64
            # print network.get_shape()
            network = tflearn.conv_2d(network, 32, 3, activation='relu', name='conv2_1')
            network = tflearn.conv_2d(network, 32, 3, activation='relu', name='conv2_2')
            network = tflearn.max_pool_2d(network, 2, strides=2)
            network = tflearn.batch_normalization(network)
            # 32 32

            network = tflearn.conv_2d(network, 64, 3, activation='relu', name='conv3_1')
            network = tflearn.conv_2d(network, 64, 3, activation='relu', name='conv3_2')

            network = tflearn.max_pool_2d(network, 2, strides=2)
            network = tflearn.batch_normalization(network)
            # 16 16

            network = tflearn.conv_2d(network, 128, 3, activation='relu', name='conv4_1')
            network = tflearn.conv_2d(network, 128, 3, activation='relu', name='conv4_2')
            # network = tflearn.conv_2d(network, 128, 3, activation='relu', name='conv4_2')

            # network = tflearn.conv_2d(network, 256, 3, activation='relu', name='conv4_3')
            #
            network = tflearn.max_pool_2d(network, 2, strides=2)
            network = tflearn.batch_normalization(network)
            # 8 8

            network = tflearn.conv_2d(network, 256, 3, activation='relu', name='conv5_1')
            network = tflearn.conv_2d(network, 256, 3, activation='relu', name='conv5_2')
            network = tflearn.max_pool_2d(network, 2, strides=2)
            network = tflearn.batch_normalization(network)
            # 4 4
        return network


    def build_RNN(self, x, reuse):

        cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hiddenDim, state_is_tuple=True, activation=tf.tanh,
                                              reuse=reuse);
        cell_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hiddenDim, state_is_tuple=True, activation=tf.tanh,
                                              reuse=reuse);
        output, _ = tf.nn.bidirectional_dynamic_rnn(cell_1, cell_2, x, dtype=tf.float32);

        output = tf.concat([output[0], output[1]], axis=2)
        output = output[:, -1]

        return output


    def build_graph(self, input, expected, reuse, batch_size):

        input = tf.reshape(input, [-1, self.input_height, self.input_width, self.input_channels])

        input = tf.cast(input, tf.float32)

        network = input / 255.0

        images = network[0:20]
        images = tf.reshape(images, [self.input_frames, self.input_height, self.input_width, self.input_channels])


        with tf.variable_scope('mainCNN', reuse=reuse):
            network = self.vggLayer(network, 'mainCNN', reuse)
            network = tf.reduce_mean(network, [1, 2])

        with tf.variable_scope('reshape', reuse=reuse):
            afterGBD = int(network.get_shape()[-1])
            network = tf.reshape(network, [-1, self.input_frames, afterGBD])
            

        with tf.variable_scope('mainRNN', reuse=reuse):
            network = self.build_RNN(network, reuse)

        with tf.variable_scope('fc_part', reuse=reuse):
            
            network = tflearn.batch_normalization(network, name='batch_fc')
            network = tflearn.fully_connected(network, 1024, name='fc1', activation=tf.nn.tanh)

        with tf.variable_scope("mainRegression", reuse=reuse):
            norm_label = expected / 20.0
            prediction, total_loss = tflearn_old.models.linear_regression(network, norm_label)
            output = prediction * 20.

        return dict(prediction=output, loss=total_loss, images=None, attention=None)

