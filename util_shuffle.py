# -*- coding: utf-8 -*-
import numpy as np
import os
import json
import threading
import time
import multiprocessing
import threading
import sys
import config
import random
import glob


class DataLoader:
    def __init__(self, data_fold_paths, data_shuffle, keep_order=False):
        """
        :param data_fold_paths:
        :param data_shuffle:
        :param keep_order:
        """
        self.data_shuffle = data_shuffle
        self.data_fold_paths = data_fold_paths
        self.keep_order = keep_order

        # initialize data arrays
        self.inputs         = None
        self.labels         = None
        self.materials      = None
        self.degrees        = None
        self.brightnesses   = None

        self.index = 0
        self.size = 0
        self.read_all_batch()

    def size(self):
        return self.size

    def shuffle(self):
        """
        Shuffle Batches
        """
        assert self.data_shuffle, "Why did you call shuffle?"
        total = list(zip(self.inputs, self.labels, self.materials, self.degrees, self.brightnesses))


        np.random.shuffle(total)

        self.inputs, self.labels, self.materials, self.degrees, self.brightnesses = zip(*total)

    def next(self, batch_size = config.BATCH_SIZE):
        assert self.size > 0, 'DataLoader.maxSize is zero. Something wrong'

        X = [None] * batch_size
        Y = [None] * batch_size
        D = [None] * batch_size
        M = [None] * batch_size
        G = [None] * batch_size

        # Collect Batches
        for i in range(batch_size):

            X[i], Y[i], M[i], D[i], G[i] = \
                self.inputs[self.index], self.labels[self.index], self.materials[self.index], self.degrees[self.index], self.brightnesses[self.index]

            self.index = (self.index + 1) % self.size

        return X, Y, M, D, G

    def read_all_batch(self):
        print '[!] Start to read all data batches'
        self.size = 0

        # Collect all paths of datasets
        paths = []
        for fold_path in self.data_fold_paths:
            wild_card = os.path.join(fold_path, '*.bin')
            paths += [file_name for file_name in glob.glob(wild_card)]
        paths.sort()



        assert len(paths) > 0 , "No paths are collected."

        # read all data informations and check validation
        for path in paths:
            with open(path, 'rb') as fin:
                dataconfig = np.load(fin)
                self.size += int(dataconfig[1]) if not config.TEST_MODE else 1
                assert config.IMAGE_FRAMES  == int(dataconfig[0]), "Wrong Image Frames in %s"   % path
                assert config.IMAGE_CHANNELS== int(dataconfig[2]), "Wrong Image Channels in %s" % path
                assert config.IMAGE_WIDTH   == int(dataconfig[3]), "Wrong Image Width in %s"    % path
                assert config.IMAGE_HEIGHT  == int(dataconfig[4]), "Wrong Image Heights in %s"  % path

        self.inputs         = np.empty([self.size, config.IMAGE_FRAMES, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS], dtype=np.uint8)
        self.labels         = np.empty([self.size], dtype=np.float32)
        self.materials      = np.empty([self.size], dtype=np.object)
        self.degrees        = np.empty([self.size], dtype=np.object)
        self.brightnesses   = np.empty([self.size], dtype=np.object)

        # create data loading workers
        self.collect_data_parallel(paths)

        print ''
        print '[!] Data Loading done. '
        print ' - Total Data Size : %d' % self.size


    def collect_data_parallel(self, paths):
        workers = [Worker(path=path) for path in paths]
        data_counter = 0
        sys.stdout.flush()
        start_time = time.time()

        if self.keep_order :
            """ KEEP ORDER MODE
            """
            for w in workers:
                while not w.empty():
                    self.inputs[data_counter], self.labels[data_counter], self.materials[data_counter], self.degrees[
                        data_counter], self.brightnesses[data_counter] = w.read()
                    data_counter += 1
                    self.print_loading_progressbar(data_counter=data_counter, timecost= time.time() - start_time)
        elif not self.keep_order:
            """ IGNORE ORDER MODE 
            """
            worker_index = 0
            while data_counter < self.size:
                w = workers[worker_index]
                if not w.empty():
                    self.inputs[data_counter], self.labels[data_counter], self.materials[data_counter], self.degrees[
                        data_counter], self.brightnesses[data_counter] = w.read()
                    data_counter += 1
                    self.print_loading_progressbar(data_counter=data_counter, timecost=time.time() - start_time)

                worker_index = (worker_index + 1) % len(workers)

        print ''
        assert data_counter == self.size, "Expected data size and loaded data size is different"

    def print_loading_progressbar(self, data_counter, timecost):
        """
        :param data_counter:
        :param timecost:
        :return:
        """
        done = int(100 * data_counter / self.size)
        remain = 100 - done
        if data_counter % 100 == 0 or remain == 0:
            sys.stdout.write(
                "\r[%s%s] %3d%% in %.3f seconds " %
                ("=" * done, " " * remain, done, timecost)
            )
            sys.stdout.flush()

    def release(self):
        self.inputs         = None
        self.labels         = None
        self.materials      = None
        self.degrees        = None
        self.brightnesses   = None

# Thread
def load_bin_file_fn(path, pipe):
    """

    :param path:
    :param pipe:
    :return:
    """
    with open(path, 'rb') as fin:
        data_config = np.load(fin)
        sample_count = int(data_config[1])

        for i in range(sample_count):
            image = np.array(np.load(fin), dtype=np.uint8)
            label = np.array(np.load(fin))
            extra_info =  np.load(fin)

            packet = [
                np.expand_dims(image,axis=3),
                label,
                extra_info[0],
                extra_info[1],
                extra_info[2]
             ]

            pipe.send(packet)

            if config.TEST_MODE:
                break
    exit(0)

class Worker:
    worker_index = 0

    def __init__(self, path):
        self.success = 0
        self.parent_pipe, self.child_pipe = multiprocessing.Pipe(duplex=False)
        self.process = multiprocessing.Process(target=load_bin_file_fn, args=(path, self.child_pipe))

        if not config.TEST_MODE:
            with open(path, 'rb') as fin:
                self.size = int(np.load(fin)[1])
        else:
            self.size = 1

        self.process.start()

    def empty(self):
        return self.success >= self.size

    def read(self):
        if self.empty():
            return None

        packet = self.parent_pipe.recv()
        self.success+=1
        return packet
