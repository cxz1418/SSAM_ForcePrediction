


# -*- coding: utf-8 -*-
import argparse
import os
import shutil

import numpy as np
import tensorflow as tf

import util_shuffle as util
import config
import time

from nets import cnn, rnn



if __name__ == '__main__':
    """
    Load all training data folds
    """
    data_manager = util.DataLoader(
        data_fold_paths = config.DATA_FOLD_PATHS,
        data_shuffle= False,
        keep_order=True
    )



    # Load batches just one epoch
    batch_index = 0
    remain_batch = data_manager.size


    label_total = []
    prediction_total = []
    mat_total = []
    deg_total = []
    bright_total = []


    while remain_batch > 0:
        batch_size = min(config.BATCH_SIZE, remain_batch)
        remain_batch -= batch_size

        # read input data from data manager
        input_batch, label_batch, mat_batch, deg_batch, bright_batch = data_manager.next(batch_size=batch_size)



        label_batch = np.array(label_batch).reshape([-1])

        label_total.append(label_batch)

        mat_total.append(np.array(mat_batch).reshape([-1]))
        deg_total.append(np.array(deg_batch).reshape([-1]))
        bright_total.append(np.array(bright_batch).reshape([-1]))

        batch_index += 1
        print batch_index

    label_total = np.array(label_total).reshape([-1])


    np.save('./labelData_train.npy', label_total)
    np.save('./matData_train.npy', mat_total)
    np.save('./degData_train.npy', deg_total)
    np.save('./brightData_train.npy', bright_total)






