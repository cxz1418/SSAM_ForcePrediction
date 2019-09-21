# -*- coding: utf-8 -*-
import argparse
import os
import shutil

import numpy as np
import tensorflow as tf

import GenerateGraph as gg
import evaluationTool as evt
import util
import config
import time
from nets import cnn, rnn

class Evaluator:
    def __init__(self):
        """
        Define your evaluator model
        """
       
        with tf.device('/cpu:0'):
            self.all_towers = []
            self.input = tf.placeholder(
                dtype=tf.float32,
                shape=[
                    None, config.IMAGE_FRAMES, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS
                ],
                name='input'
            )

            self.label = tf.placeholder(dtype=tf.float32, shape=[None], name='label')

            # build towers for each GPUs
            # for i in range(config.NUM_GPUS):
            self.build_tower(0, self.input, self.label)
            self.output = self.all_towers[0]['prediction']

            # self.batch_outputs = tf.concat(self.tower_outputs, axis=-1)
        

    def build_tower(self, gpu_index, X, Y):
        
        with tf.device('/gpu:%d' % gpu_index), tf.name_scope('tower_%d' % gpu_index), tf.variable_scope(
                tf.get_variable_scope(), reuse=gpu_index is not 0):
            graph = config.MODEL_NETWORK.build_graph(X, Y, gpu_index is not 0,-1)
            output = graph['prediction']

            self.all_towers.append(graph)
            # self.tower_outputs.append(output)
            tf.get_variable_scope().reuse_variables()

    def run_evaluate(self, data_manager, model_save_path, result_save_path):
        """
        blar blar
        """
        session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        eval_start_time = time.time()

        evaluation_all = evt.EvaluationMatrixGenerator()
        evaluation_paper = evt.EvaluationMatrixGenerator()
        evaluation_sponge = evt.EvaluationMatrixGenerator()
        evaluation_stapler = evt.EvaluationMatrixGenerator()
        evaluation_tube = evt.EvaluationMatrixGenerator()

        with tf.device('/cpu:0'), session.as_default():
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            ckpt  = tf.train.get_checkpoint_state(model_save_path)

            assert ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path), "Model doesn't exist - %s " % model_save_path
            #'saver.restore(session, '/mnt/hochul/TrainSave_sam/RCNN_0806_MultiModal_Attention_Parallel/tr-266500')'
            #saver.restore(session, '/mnt/hochul/TrainSave_sam/RCNN_0806_MultiModal_Attention_Parallel/tr-259500')

            ##############/mnt/hochul/TrainSave_sam/RCNN_0819_MultiModal_Attention_CH_newM_relu_sum/tr-111750
            # saver.restore(session,'/mnt/hochul/TrainSave_sam/RCNN_0819_MultiModal_Attention_CH_newM_relu_sum/tr-111750')
            #saver.restore(session,'/mnt/hochul/TrainSave_sam/RCNN_0809_MultiModal_Attention_series_CH_SP_parallel_add/tr-259260')


            saver.restore(session, ckpt.model_checkpoint_path)

            #saver.restore(session,'/mnt/hochul/TrainSave_sam/RCNN_1015_sigmoid_Res_Spatial/tr-458175')
            #saver.restore(session,'/mnt/hochul/TrainSave_sam/RCNN_0824_3_Parallel_1x1_residual_fixRelu/tr-87165')




            shutil.rmtree(result_save_path, ignore_errors=True)
            os.makedirs(result_save_path)
            os.makedirs(os.path.join(result_save_path, 'figures'))

            remain_batch = data_manager.size

            # Load batches just one epoch
            batch_index = 0
            while remain_batch > 0:
                batch_size = min(config.BATCH_SIZE, remain_batch)
                remain_batch -= batch_size

                # read input data from data manager
                input_batch, label_batch, mat_batch, deg_batch, bright_batch = data_manager.next(batch_size=batch_size)

                prediction_batch = session.run(self.output, feed_dict={self.input: input_batch, self.label: label_batch})





                label_batch = np.array(label_batch).reshape([-1])
                prediction_batch = np.array(prediction_batch).reshape([-1])

                fullpath= '{0}_{1}_{2}'.format(mat_batch[0],deg_batch[0],bright_batch[0])
                fullpath = str(fullpath)
                #print fullpath
                if config.EVAL_SAVE_FIGURES:

                    gg.GenerateGraph(prediction_batch, label_batch, os.path.join(result_save_path, 'figures'), fullpath+str(batch_index), False)

                evaluation_all.addTable(mat_batch, deg_batch, bright_batch, label_batch, prediction_batch, 'None')
                evaluation_paper.addTable(mat_batch, deg_batch, bright_batch, label_batch, prediction_batch, 'papercup')
                evaluation_sponge.addTable(mat_batch, deg_batch, bright_batch, label_batch, prediction_batch, 'sponge')
                evaluation_stapler.addTable(mat_batch, deg_batch, bright_batch, label_batch, prediction_batch, 'stapler')
                evaluation_tube.addTable(mat_batch, deg_batch, bright_batch, label_batch, prediction_batch, 'tube')


                batch_index += 1

          
            evaluation_all.saveTables(result_save_path, 'total')
            evaluation_paper.saveTables(result_save_path, 'papercup')
            evaluation_sponge.saveTables(result_save_path, 'sponge')
            evaluation_stapler.saveTables(result_save_path, 'stapler')
            evaluation_tube.saveTables(result_save_path, 'tube')

#
# def Evaluation(data_manager, result_save_path):
#     # ConfigData from 'config' object
#     num_gpus = config['environment']['num_gpus']
#     learning_rate = config['train']['learning_rate']
#     num_steps = config['train']['num_steps']
#     save_steps = config['train']['save_interval']
#
#     image_width = config['data']['image_width']
#     image_height = config['data']['image_height']
#     image_channels = config['data']['image_channels']
#
#     num_frames = config['data']['num_frames']
#     batch_size = config['data']['batch_size']
#
#
#     result_path     =  config['eval']['result_path'] + '/' + config['eval']['result_name']
#     model_save_path      = config['environment']['save_folder'] + '/' + config['environment']['save_name']
#
#
#     makeResultFold(result_path)
#
#     # total image size
#     image_size = image_width * image_height * image_channels
#
#     with tf.device('/cpu:0'):
#         global_step = tf.Variable(0, trainable=False, name='global_step')
#         increment_global_step = tf.assign_add(global_step, 1, name='increment_global_step')
#
#         data = tf.placeholder(dtype=tf.float32, shape=[None, num_frames, image_size], name='data')
#         label = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')
#
#         graph = model.build_graph(data, label, False, num_frames, image_height, image_width,
#                                   image_channels)
#
#         prediction = graph['prediction']
#
#         init = tf.global_variables_initializer()
#
#     config_proto = tf.ConfigProto(allow_soft_placement=True)
#     sess = tf.Session(config=config_proto)
#     sess.run(init)
#
#     print 'start prediction'
#     with sess.as_default():
#
#         ########Save checkPoint to TrainD#########
#         saver = tf.train.Saver(max_to_keep=90)
#         ckpt = tf.train.get_checkpoint_state(model_save_path)
#         if ckpt and ckpt.model_checkpoint_path:
#             print ckpt.model_checkpoint_path
#             saver.restore(sess, ckpt.model_checkpoint_path)
#         else:
#             os.makedirs(model_save_path);
#         ##########################################
#
#
#         evaluationManager = evt.EvaluationMatrixGenerator()
#         evaluation_paper =  evt.EvaluationMatrixGenerator()
#         evaluation_sponge= evt.EvaluationMatrixGenerator()
#         evaluation_stapler = evt.EvaluationMatrixGenerator()
#         evaluation_tube = evt.EvaluationMatrixGenerator()
#
#         print '--------  %d   ---------'%(data_manager.maxSize/data_manager.batchSize)
#         for s in range(data_manager.maxSize/data_manager.batchSize):
#             data_batch, label_batch, material, degree, bright = data_manager.next()
#
#             inputs = {data: data_batch, label: label_batch}
#             results = sess.run([prediction], feed_dict=inputs)
#
#             predictionR = np.copy(results)
#             predictionR = np.reshape(predictionR, (-1))
#             groundTruth = np.copy(inputs[label])
#             groundTruth = np.reshape(groundTruth, (-1))
#
#             if config.EVAL_SAVE_FIGURES:
#                 gg.GenerateGraph(predictionR, groundTruth, result_path + '/graphResult', 'Sample_%d' % s, False)
#
#             evaluationManager.addTable(material, degree, bright, groundTruth, predictionR,'None')
#
#             evaluation_paper.addTable(material, degree, bright, groundTruth, predictionR, 'papercup')
#             evaluation_sponge.addTable(material, degree, bright, groundTruth, predictionR, 'sponge')
#             evaluation_stapler.addTable(material, degree, bright, groundTruth, predictionR, 'stapler')
#             evaluation_tube.addTable(material, degree, bright, groundTruth, predictionR, 'tube')
#
#
#                 # evaluationManager.addTable(material, degree, bright, groundTruth, predictionR)
#                 # evaluationManager_each.addTable(material, degree, bright, groundTruth, predictionR)
#
#         print result_path
#         evaluationManager.saveTables(result_path, 'total', evaluationManager.sampleNum)
#
#         evaluation_paper.saveTables(result_path, 'papercup')
#         evaluation_sponge.saveTables(result_path, 'sponge')
#         evaluation_stapler.saveTables(result_path, 'stapler')
#         evaluation_tube.saveTables(result_path, 'tube')
#


"""
MAIN SECTION
"""
if __name__ =='__main__':
    """
    Load all evaluation data folds
    """

    print config.EVAL_FOLD_PATHS
    data_manager = util.DataLoader(
        data_fold_paths= config.EVAL_FOLD_PATHS,
        data_shuffle=False,
        keep_order=True
    )

    evaluator = Evaluator()
    evaluator.run_evaluate(
        data_manager=data_manager,
        model_save_path= os.path.join(config.MODEL_SAVE_FOLDER, config.MODEL_SAVE_NAME),
        result_save_path=os.path.join(config.EVAL_SAVE_FOLDER, config.MODEL_SAVE_NAME)
    )

    # #read default configuration options from 'config.json'
    # config = util.load_config()
    #
    # print 'loading DB'
    #
    # # declare your own model object
    # model = getattr(models, config['environment']['model_name'])()
    # # config['eval']['result_name']
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str)
    # arg = parser.parse_args()
    #
    # if arg.model_name :
    #     config['environment']['save_name'] = arg.model_name
    #     config['eval']['result_name'] = arg.model_name
    #
    # #load all test dataset from multiple data folders into array
    # data_path = config['train']['data_path']
    # batchGenerator = util.DataLoader(data_path, False, 256, 4, 'test', test_mode=config['train']['test_mode'])
    #
    #
    # Evaluation(batchGenerator, model, config)

