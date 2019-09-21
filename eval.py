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
            self.build_tower(0, self.input, self.label)
            self.output = self.all_towers[0]['prediction']


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

            saver.restore(session, ckpt.model_checkpoint_path)

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
