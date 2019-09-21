# -*- coding: utf-8 -*-
from nets.common import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import layers as tflayers
from tensorflow.contrib import learn as tflearn_old
import tflearn


class RCNN_0718_BaseLine(BaseModel):
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



            '''
            network = tflearn.conv_2d(network, 128, 3, activation='relu', name='conv5_1')
            #network = tflearn.conv_2d(network, 128, 3, activation='relu', name='conv5_2')
            #network = tflearn.conv_2d(network, 256, 3, activation='relu', name='conv5_3')

            network = tflearn.max_pool_2d(network, 2, strides=2)
            network = tflearn.batch_normalization(network)
            #4 4
            '''
        return network

    def build_attention_matrix_v2_soft(self, input, reuse, depth=2, init_channel=16):

        input = self.vggLayer(input, 'vgg_att_soft', reuse)

        with tf.variable_scope('attention', reuse=reuse):
            network = tflearn.conv_2d(input, 1, 3, activation='linear', name='att_fwd', bias=True, bias_init='xavier')

            network = tf.exp(network)
            total = tf.reduce_sum(network, [1, 2])
            total = tf.expand_dims(total, axis=1)
            total = tf.expand_dims(total, axis=1)
            network = network / total

            network = tf.image.resize_images(network, size=[self.input_width, self.input_height],
                                             method=tf.image.ResizeMethod.BILINEAR)

            # network = tf.expand_dims(network,axis=3)

            return network

    def build_multimodal_attention(self, input, reuse):

        with tf.variable_scope('attention', reuse=reuse):
            network = tflearn.conv_2d(input, 128, 3, activation='relu', name='att_1', bias=True, bias_init='xavier')
            network = tflearn.conv_2d(network, 1, 3, activation='linear', name='att_2', bias=True, bias_init='xavier')

            
            width = int(network.get_shape()[1])
            height = int(network.get_shape()[2])
            network = tf.reshape(network, [-1, width * height])
           

            network = tf.nn.softmax(network)
            
            network = tf.reshape(network, [-1, width, height, 1])
          

            # network = tf.expand_dims(network,axis=3)

            return network

    def attention(self, reuse, inputs, attention_size, time_major=False, return_alphas=False):

      

        hidden_size = self.hiddenDim  # D value - hidden size of the RNN layer

        # encode_image_W = tf.get_variable("encode_image_W", [dim_cnn, self.hiddenDim],
        #                                  initializer=tf.random_normal_initializer())

        with tf.variable_scope("variables", reuse=reuse):
            w_omega = tf.get_variable("w_omega", [hidden_size * 2, attention_size]
                                      , initializer=tf.random_normal_initializer())
            b_omega = tf.get_variable("b_omega", [attention_size]
                                      , initializer=tf.random_normal_initializer())
            u_omega = tf.get_variable("u_omega", [attention_size]
                                      , initializer=tf.random_normal_initializer())

        v = tf.reshape(inputs, [-1, hidden_size * 2])
        v = tf.matmul(v, w_omega)
        v = tf.tanh(v + b_omega)

      

        vu = v * u_omega
        vu = tf.reshape(vu, [-1, self.input_frames, attention_size])
        vu = tf.reduce_sum(vu, 2)

       

        #
        # print '-------k--------'
        # print inputs.get_shape()
        #
        # with tf.variable_scope('v',reuse = reuse):
        #     v = tf.tanh(tf.tensordot(inputs, w_omega, axes=2) + b_omega)
        #
        # #tf.nn.xw_plus_b(video_flat, encode_image_W, encode_image_b)
        # print '-----v------'
        # print v.get_shape()
        #
        # vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape

       
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

       

        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        if not return_alphas:
            return output
        else:
            return output, alphas

    def RNN_att_map(self, x, reuse, dim_cnn, batch_size):

        with tf.variable_scope("variables", reuse=reuse):

            lstm_att = tf.nn.rnn_cell.BasicLSTMCell(self.hiddenDim, self.hiddenDim * 2, state_is_tuple=False,
                                                    reuse=reuse)

            encode_image_W = tf.get_variable("encode_image_W", [dim_cnn, self.hiddenDim],
                                             initializer=tf.random_normal_initializer())
            encode_image_b = tf.get_variable("encode_image_b", [self.hiddenDim],
                                             initializer=tf.zeros_initializer())

            embed_att_Wa = tf.get_variable("embed_att_Wa", [self.hiddenDim, self.hiddenDim],
                                           initializer=tf.random_normal_initializer())

            embed_att_Ua = tf.get_variable("embed_att_Ua", [self.hiddenDim, self.hiddenDim],
                                           initializer=tf.random_normal_initializer())

            embed_att_ba = tf.get_variable("encode_att_ba", [self.hiddenDim],
                                           initializer=tf.zeros_initializer())

            current_embed = tf.get_variable("current_embed", [batch_size, self.hiddenDim],
                                            initializer=tf.zeros_initializer())

            state1 = tf.get_variable("state1", [batch_size, self.hiddenDim * 2],
                                     initializer=tf.zeros_initializer())

            h_prev = tf.get_variable("h_prev", [batch_size, self.hiddenDim],
                                     initializer=tf.zeros_initializer())

        # dim_cnn : last dim of cnn_net
        video_flat = tf.reshape(x, [batch_size * self.input_frames, dim_cnn])  # (b x n) x d
        image_emb = tf.nn.xw_plus_b(video_flat, encode_image_W, encode_image_b)  # (b x n) x h

        image_emb = tf.reshape(image_emb, [batch_size, self.input_frames, self.hiddenDim])  # b x n x h
        image_emb = tf.transpose(image_emb, [1, 0, 2])  # n x b x h

        image_part = tf.matmul(image_emb, tf.tile(tf.expand_dims(embed_att_Ua, 0),
                                                  [self.input_frames, 1, 1])) + embed_att_ba  # n x b x h

        for i in range(self.input_frames):
            e = tf.tanh(tf.matmul(h_prev, embed_att_Wa) + image_part)  # n x b x h
            e = tf.reduce_sum(e, 2)  # n x b
            denomin = tf.reduce_sum(e, 0)  # b
            denomin = denomin + tf.to_float(tf.equal(denomin, 0))  # regularize denominator
            alphas = tf.tile(tf.expand_dims(tf.div(e, denomin), 2),
                             [1, 1, self.hiddenDim])  # n x b x h  # normalize to obtain alpha

            attention_list = alphas * image_emb  # n x b x h
            atten = tf.reduce_sum(attention_list, 0)  # b x h       #  soft-attention weighted sum
            if i > 0: tf.get_variable_scope().reuse_variables()

           

            with tf.variable_scope("lstm_att", reuse=reuse):
                output, state1 = lstm_att(tf.concat([atten, current_embed], 1), state1)  # b x h
           

            h_prev = output  # b x h

        return output

    def build_RNN(self, x, reuse):

        cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hiddenDim, state_is_tuple=True, activation=tf.tanh,
                                              reuse=reuse);
        cell_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hiddenDim, state_is_tuple=True, activation=tf.tanh,
                                              reuse=reuse);
        output, _ = tf.nn.bidirectional_dynamic_rnn(cell_1, cell_2, x, dtype=tf.float32);

        output = tf.concat([output[0], output[1]], axis=2)
        output = output[:, -1]

       
        return output

    def localRNN(self, x, reuse, cell_1, cell_2, _scope):

        output, _ = tf.nn.bidirectional_dynamic_rnn(cell_1, cell_2, x, dtype=tf.float32, scope=_scope);

        output = tf.concat([output[0], output[1]], axis=2)
        output = output[:, -1]

        # self attention or using last gruCell Output

        return output

    # Because the time step not matched correctely
    # so there are some strange vector length
    # Check CAUTION comment

    def SliceDB(self, x, sliceFrameNum):

        x = tf.slice(x, [0, self.input_frames - sliceFrameNum, 0], [-1, sliceFrameNum, -1])
        return x

    def RLC(self, filter_name, x, hiddenDim, activation, reuse, stride=1, filterSize=3, channels=1, padding='None'):
        # x : input rnn vector ( maybe [batch,timeStep,vectorDim)
        # stride : filter stride
        # nb_filter : num of channels
        # padding ; 'None' or ZeroPadding'

        timeStep = x.get_shape()[1]

        if (timeStep < filterSize):
            ErrorLogAndExit("initLen > filterSize")

        # timeStep Length

        if (((timeStep - filterSize) % (stride) != 0 and padding == 'None') or
                ((timeStep) % (stride) != 0 and padding == 'ZeroPadding')):
            ErrorLogAndExit("Please Fit Timestep Size : Required TimeStep(filterSize+stride*(k-1) //k=nexeTimeStep\n"
                            + "Please Fit Timestep Size(ZeroPadding) : Required TimeStep(filterSize+stride*(k) //k=nexeTimeStep")

        with tf.variable_scope(filter_name, reuse=reuse):
            cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hiddenDim, state_is_tuple=True,
                                                  activation=activation,
                                                  reuse=reuse);
            cell_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hiddenDim, state_is_tuple=True,
                                                  activation=activation,
                                                  reuse=reuse);

        if padding == 'None':
            nextTimeStepNum = (timeStep - filterSize) / stride + 1
            for i in range(nextTimeStepNum):
                
                sliceVector = tf.slice(x, [0, i * stride, 0], [-1, filterSize, -1])
                if (i == 0):

                    self.nextLayer = self.localRNN(sliceVector, True, cell_1, cell_2, filter_name)
                else:
                    self.nextLayer = tf.concat([self.nextLayer, self.localRNN(sliceVector, True, cell_1, cell_2,
                                                                              filter_name)], 1)

            x = tf.reshape(self.nextLayer, [-1, int(nextTimeStepNum), hiddenDim * 2])

        elif (padding == 'ZeroPadding'):

            startX = -(filterSize - 1) / 2
            endX = startX + filterSize
            nextTimeStepNum = (timeStep) / stride

            for i in range(nextTimeStepNum):

                if (startX < 0):
                    sliceSt = 0
                    sliceNum = filterSize + startX
                elif (endX >= timeStep):
                    sliceSt = startX
                    sliceNum = timeStep - sliceSt
                else:
                    sliceSt = startX
                    sliceNum = filterSize

                # sliceSt= int(sliceSt)
                sliceNum = int(sliceNum)

               
                sliceVector = tf.slice(x, [0, sliceSt, 0], [-1, sliceNum, -1])

                if (i == 0):

                    self.nextLayer = self.localRNN(sliceVector, True, cell_1, cell_2, filter_name)
                else:
                    self.nextLayer = tf.concat([self.nextLayer, self.localRNN(sliceVector, True, cell_1, cell_2,
                                                                              filter_name)], 1)
                startX += stride
                endX += stride

            x = tf.reshape(self.nextLayer, [-1, int(nextTimeStepNum), hiddenDim * 2])
        return x

    def ConvGRU_MakaBaseLine(self, x, reuse):

        input = self.SliceDB(x, 15)

        # time step =15
        input = self.RLC('depth_1', input, 128, tf.tanh, reuse, stride=1, filterSize=3, channels=1,
                         padding='ZeroPadding')
        input = self.RLC('depth_1_pooling', input, 128, tf.tanh, reuse, stride=2, filterSize=3, channels=1,
                         padding='None')
        # actually it is stride pooling


        # time step =7
        input = self.RLC('depth_2', input, 128, tf.tanh, reuse, stride=1, filterSize=3, channels=1,
                         padding='ZeroPadding')
        input = self.RLC('depth_2_pooling', input, 128, tf.tanh, reuse, stride=2, filterSize=3, channels=1,
                         padding='None')

        # time step=3
        input = self.RLC('depth_3', input, 128, tf.tanh, reuse, stride=1, filterSize=3, channels=1,
                         padding='ZeroPadding')
        input = self.RLC('depth_3_pooling', input, 128, tf.tanh, reuse, stride=2, filterSize=3, channels=1,
                         padding='None')

        return tf.reshape(input, [-1, int(x.get_shape()[2])])

    def ConvGRU(self, x, reuse):
       
        filterSize = 3;
        stride = 2;

        # curBatchSize = int(x.get_shape()[0])
        vectorDim = int(x.get_shape()[2])
        initLen = int(x.get_shape()[1])

        ########CAUTION Only for TimeStep-20###############
        if (initLen < filterSize or (initLen - filterSize) % 2 != 0):
            ############CAUTION
            x = tf.slice(x, [0, 5, 0], [-1, 15, -1])
            initLen = int(x.get_shape()[1])

           

            ############CAUTION

            # print 'impossible timestep with filterSize : %d and stride : %d'%(filterSize,stride)
            # exit()

        while (True):
            if (initLen < filterSize):
                break;

            
            forNum = (initLen - filterSize) / stride + 1
            with tf.variable_scope("gru_layer2_" + str(initLen), reuse=reuse):

                cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hiddenDim, state_is_tuple=True, activation=tf.tanh,
                                                      reuse=reuse);

                cell_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hiddenDim, state_is_tuple=True, activation=tf.tanh,
                                                      reuse=reuse);
     
            for i in range(forNum):
                
                sliceVector = tf.slice(x, [0, i * stride, 0], [-1, filterSize, -1])
                

                if (i == 0):
                    self.nextLayer = self.localRNN(sliceVector, True, cell_1, cell_2, "gru_layer2_" + str(initLen))
                else:
                    self.nextLayer = tf.concat([self.nextLayer, self.localRNN(sliceVector, True, cell_1, cell_2,
                                                                              "gru_layer2_" + str(initLen))], 1)

            ###concat
            vectorDim = self.hiddenDim * 2
            x = tf.reshape(self.nextLayer, [-1, forNum, vectorDim])

            initLen = int(initLen / stride)

        afterDim = int(x.get_shape()[2])
        return tf.reshape(x, [-1, forNum * afterDim])

    def build_graph(self, input, expected, reuse, batch_size):

        input = tf.reshape(input, [-1, self.input_height, self.input_width, self.input_channels])

        input = tf.cast(input, tf.float32)

        network = input / 255.0

        images = network[0:20]
        images = tf.reshape(images, [self.input_frames, self.input_height, self.input_width, self.input_channels])

        # print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
        # print images.get_shape()


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

