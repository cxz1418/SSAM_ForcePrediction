


# -*- coding: utf-8 -*-
import argparse
import os
import shutil

import numpy as np
import tensorflow as tf

import util
import config
import time

from nets import cnn, rnn
import sys

def average_gradients(grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def getCurrentTotalNum():
    totalNum =0
    for k in range(len(config.DATA_FOLD_NUM)):
        if(k==config.EVAL_FOLD_INDEX):
            continue
        else:
            totalNum+=config.DATA_FOLD_NUM[k]

    return totalNum





class Trainer:
    def __init__(self):
        """
        Define your variables and training structure
        """
        print ('[!] START INITIALIZING TRAINER')
        with tf.device('/cpu:0'):
            # global variables from all towers
            self.all_towers   =  []
            self.all_outputs  =  []
            self.all_gradients = []
            self.all_losses =    []

            # basic variables
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
            #self.optimizer = config.MODEL_OPTIMIZER(learning_rate=self.learning_rate,momentum=0.9)
            self.optimizer = config.MODEL_OPTIMIZER(learning_rate=self.learning_rate)


            #Load data from tfrecords
            feature = {
                'frames': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'channels': tf.FixedLenFeature([], tf.int64),
                'force': tf.FixedLenFeature([], tf.float32),
                'material': tf.FixedLenFeature([], tf.string),
                'degree': tf.FixedLenFeature([], tf.string),
                'bright': tf.FixedLenFeature([], tf.string),
                'video': tf.FixedLenFeature([], tf.string)

            }
            # Create a list of filenames and pass it to a queue



            foldPaths = []
            for qw in range(10):
                path = '/mnt/hochul/TrainSave_sam/TFRecord/train/trainset.tfrecords%d' % qw
                foldPaths.append(path)
            # Create a list of filenames and pass it to a queue
            filename_queue = tf.train.string_input_producer(
                foldPaths
                , num_epochs=None)
            # Define a reader and read the next record

            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            # Decode the record read by the reader
            features = tf.parse_single_example(serialized_example, features=feature)
            # Convert the image data from string back to the numbers


            force = tf.cast(features['force'], tf.float32)
            # force =  force.decode("utf-8")
            print ('----------------')
            print (force.get_shape())

            video = tf.decode_raw(features['video'], tf.uint8)

            print( '----------------')
            print (video.get_shape())

            # Reshape image data into the original shape
            video = tf.reshape(video,
                               [config.IMAGE_FRAMES,config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.IMAGE_CHANNELS])

            print ('----------------')
            print (video.get_shape())
            print (force.get_shape())

            video_batch,force_batch =tf.train.shuffle_batch([video, force], batch_size=config.BATCH_SIZE,
                                   capacity=config.BATCH_SIZE * 20, num_threads=4,
                                   min_after_dequeue=5)

            # input and label data for each towers
            self.tower_inputs = tf.split(video_batch, config.NUM_GPUS, 0)
            self.tower_labels = tf.split(force_batch, config.NUM_GPUS, 0)

            # build towers for each GPUs
            for i in range(config.NUM_GPUS):
                self.build_tower(i, self.tower_inputs[i], self.tower_labels[i])

            # define your training step
            self.train_step = tf.group(
                self.optimizer.apply_gradients(
                    global_step=self.global_step,
                    grads_and_vars= average_gradients(self.all_gradients)
                )
            )

            # define statistical information
            self.global_loss = tf.reduce_mean(self.all_losses)
            self.global_mse = tf.reduce_mean(
                tf.squared_difference(
                    tf.concat(self.all_outputs,  axis=-1),  # all outputs
                    tf.concat(self.tower_labels, axis=-1)   # all labels
                )
            )
            self.summary = self.define_summarizer()
        print ('[!] INITIALIZING DONE')




    def define_summarizer(self):
        """
        Define summary information for tensorboard
        """
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.scalar('global_loss', self.global_loss)
        tf.summary.scalar('global_mse', self.global_mse)

        # # if (self.all_towers[0]['fwd_images'] != None):
        # #     tf.summary.image('fwd_image', self.all_towers[0]['fwd_images'], 5)
        #
        # if (self.all_towers[0]['fwd_attention'] != None):
        #     tf.summary.image('fwd_attention', self.all_towers[0]['fwd_attention'], 5)
        #
        # # if (self.all_towers[0]['mid_images'] != None):
        # #     tf.summary.image('mid_image', self.all_towers[0]['mid_images'], 5)
        #
        # if (self.all_towers[0]['mid_attention'] != None):
        #     tf.summary.image('mid_attention', self.all_towers[0]['mid_attention'], 5)
        #
        # if (self.all_towers[0]['bwd_images'] != None):
        #     tf.summary.image('bwd_images', self.all_towers[0]['bwd_images'], 5)
        #
        # if (self.all_towers[0]['bwd_attention'] != None):
        #     tf.summary.image('bwd_attention', self.all_towers[0]['bwd_attention'], 5)
       

        return tf.summary.merge_all()




    def build_tower(self, gpu_index, X, Y):
        print ('[!] BUILD TOWER %d' % gpu_index)
        with tf.device('/gpu:%d' % gpu_index), tf.name_scope('tower_%d' % gpu_index), tf.variable_scope(tf.get_variable_scope(), reuse= gpu_index is not 0):
            graph = config.MODEL_NETWORK.build_graph(X, Y, gpu_index is not 0,config.BATCH_SIZE/config.NUM_GPUS )
            loss = graph['loss']
            output = graph['prediction']
            gradients = self.optimizer.compute_gradients(loss)

            self.all_towers.append(graph)
            self.all_losses.append(loss)
            self.all_outputs.append(output)
            self.all_gradients.append(gradients)
            tf.get_variable_scope().reuse_variables()


    def get_learning_rate(self, step_index=0):
        """
        Define your own learning rate strategy here
        """
        init_learning_rate  = float(config.TRAIN_LEARNING_RATE)
        decay               = float(config.TRAIN_DECAY)
        decay_level         = int(step_index/config.TRAIN_DECAY_INTERVAL)

        if isinstance(config.MODEL_NETWORK, cnn.ResNet3D):
            return init_learning_rate * (decay ** decay_level)
            # blar blar
        elif isinstance(config.MODEL_NETWORK, cnn.ResNet3D):
            return init_learning_rate * (decay ** decay_level)
            # blar blar
        else:
            return init_learning_rate * (decay ** decay_level)


    def run_train(self, model_save_path):
        """
        blar blar
        """
        session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))



        train_start_time = time.time()
        with tf.device('/cpu:0'), session.as_default():
            """
            Configurations for saving and loading your model
            """
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=1200)
            ckpt  = tf.train.get_checkpoint_state(model_save_path)
            if ckpt and ckpt.model_checkpoint_path and not config.TRAIN_OVERWRITE:
                print ('Restore trained model from ' + ckpt.model_checkpoint_path)
                print (ckpt.model_checkpoint_path)
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print ('Create new model and overwrite previous files')
                shutil.rmtree(model_save_path)
                os.makedirs(model_save_path)
                session.run(tf.global_variables_initializer())

            summary_writer = tf.summary.FileWriter(os.path.join(model_save_path, 'logs'))

            """
            Train your model iteratively
            """

            epoch = 0
            loss_array = []
            last_loss=999999
            lr_down_level=0
            weight_decay =float(config.TRAIN_WEIGHT_DECAY)
            weight_decay_level = 0
            init_lr= float(config.TRAIN_LEARNING_RATE)

            ##
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            ##



            while session.run(self.global_step) < config.TRAIN_MAX_STEPS:

                lr = init_lr*(config.TRAIN_DECAY **lr_down_level)
                lr -= lr*weight_decay*weight_decay_level

                step_index = int(session.run(self.global_step))


                step_start_time = time.time()


                # read input data from data manager



                _, loss, mse, summary = session.run(
                    fetches=[
                        self.train_step,
                        self.global_loss,
                        self.global_mse,
                        self.summary
                    ],
                    feed_dict={
                        self.learning_rate: lr
                    }
                )

                # print useful logs in your console
                timecost = time.time() - step_start_time
                if(step_index%500==0):
                    print ('[Step %5d] LR: %.5E, LOSS: %.5E, MSE: %.5E, Time: %.7f sec' % ( step_index, lr, loss, mse, timecost))

                # Save your summary information for tensor-board and model data
                summary_writer.add_summary(summary, global_step=step_index)

                if step_index % config.MODEL_SAVE_INTERVAL == 0:
                    saver.save(session, os.path.join(model_save_path, 'tr'), global_step=step_index)


                loss_array.append(loss)
                if (step_index % ( getCurrentTotalNum()/config.BATCH_SIZE) == 0 and step_index!=0):
                    epoch = int(step_index / (getCurrentTotalNum()/ config.BATCH_SIZE))
                    print ('current epoch : %d' % epoch)

                    cur_loss= np.average(loss_array)
                    loss_array=[]

                    print ('cur loss : %f , last loss : %f'%(cur_loss,last_loss))
                    if(epoch==30 or epoch ==60 or epoch==90):
                    #if(cur_loss>last_loss): #not be better
                        lr_down_level+=1
                        print ('lr_down_level up : %d' % lr_down_level)
                        weight_decay_level=0
                    else:
                        weight_decay_level +=1
                        print ('weight decay level up : %d' % weight_decay_level)
                    if (epoch == 120):
                        exit(-1);
                    last_loss=cur_loss



                # if it's test mode, stop training
                if config.TEST_MODE :
                    print ('--- All Test Done Successfully ----')
                    exit(-1)


        coord.request_stop()
        coord.join(threads)
        session.close()
        train_end_time = time.time()
        print ('--- All Training Done Successfully in %.7f seconds ----' % (train_end_time - train_start_time))



"""
MAIN SECTION
"""
if __name__ == '__main__':




    """
    Init trainer and run training sessions
    """
    trainer = Trainer()
    trainer.run_train(
        model_save_path = os.path.join(config.MODEL_SAVE_FOLDER, config.MODEL_SAVE_NAME)
    )

