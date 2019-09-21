# -*- coding: utf-8 -*-
from nets.common import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers
import tflearn as TFLEARN


"""     
****************************************************************************************************************************
"""
class RNNModel(BaseModel):


    def build_graph(self,  input, expected, reuse):
        # _x,_timeStep,_reuse,_vectorSize

        hidden_dim = 512

        cell_1= tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim,state_is_tuple=True,activation=tf.tanh,reuse= reuse);
        cell_2= tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim,state_is_tuple=True,activation=tf.tanh,reuse= reuse);
        output,_=tf.nn.bidirectional_dynamic_rnn(cell_1,cell_2, input,dtype=tf.float32);

        # reduce mean
        output = tf.concat([output[0], output[1]], axis=2)
        output = output[:, -1]


        #fc part
        output = TFLEARN.batch_normalization(output, scope='batch_fc1',reuse=reuse)
        output = tflayers.fully_connected(output, 512, activation_fn=tf.nn.relu, scope='fc/fc_1', reuse=reuse)
        output = tf.nn.dropout(output, keep_prob=0.9)
        output = TFLEARN.batch_normalization(output, scope='batch_fc2',reuse=reuse)
        output = tflayers.fully_connected(output,512,activation_fn=tf.nn.relu,scope='fc/fc_2',reuse=reuse)
        output = tf.nn.dropout(output,keep_prob=0.9)
        output = tflayers.fully_connected(output, 1024, activation_fn=tf.nn.sigmoid, scope='fc/fc_3', reuse=reuse)


        with tf.variable_scope("linear_part", reuse=reuse):
            prediction, loss = tflearn.models.linear_regression(output, expected)

        return dict(loss=loss, prediction=prediction)



class RCNNModel(BaseModel):
    def build_RNN(self,_x, _timeStep, _reuse, _vectorSize):

        hidden_dim = 512

        cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh,
                                              reuse=_reuse);
        cell_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh,
                                              reuse=_reuse);
        output, _ = tf.nn.bidirectional_dynamic_rnn(cell_1, cell_2, _x, dtype=tf.float32);

        # reduce mean
        output = tf.concat([output[0], output[1]], axis=2)
        output = output[:, -1]

        return output

    def build_vggLayer(self,_input, _step, _name, _iChannel, _oChannel, _willBN):

        iChannel = _iChannel
        output = _input

        if (_willBN == True):
            output = TFLEARN.batch_normalization(output, name='batch_' + _name)

        for i in range(_step):
            weight = tf.get_variable("w_%d_" % (i) + _name, shape=[3, 3, iChannel, _oChannel],
                                     initializer=tflayers.xavier_initializer());
            output = tf.nn.conv2d(output, weight, strides=[1, 1, 1, 1], padding='SAME');
            iChannel = _oChannel

        output = tf.nn.relu(output);
        output = tf.nn.avg_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME');  # [-1,12,12,16]

        return output

    def build_CNN(self,_x, _reuse, _h, _w, _c, _t):


        with tf.variable_scope("conv_part1", reuse=_reuse):
            layer1 = self.build_vggLayer(tf.reshape(_x, [-1, _h, _w, _c]), 1, 'l1', _c, 4, False)
            layer2 = self.build_vggLayer(layer1, 1, 'l2', 4, 16, True)
            layer2 = self.build_vggLayer(layer2, 1, 'l2_2', 16, 64, True)
            layer3 = self.build_vggLayer(layer2, 1, 'l3', 64, 128, True)
            layer4 = self.build_vggLayer(layer3, 1, 'l4', 128, 256, True)

        output = tf.reshape(layer4, [-1, _t, 3 * 3 * 256])


        return output

    def build_graph(self,_input, _expected, _reuse, _timeStep, _height, _width, _channels):

        # CNN Process
        output = self.build_CNN(_input, _reuse, _width, _height, _channels, _timeStep)

        # RNN Process
        output = self.build_RNN(output, _timeStep, _reuse, _width * _height * _channels)

        # Regression Process
        with tf.variable_scope("linear_part", reuse=_reuse):
            prediction, loss = tflearn.models.linear_regression(output, _expected)

        return dict(loss=loss, prediction=prediction)

