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

result_save_path=os.path.join(config.EVAL_SAVE_FOLDER, 'spatial_channel_ensemple_(sp_ch_frame2')
shutil.rmtree(result_save_path, ignore_errors=True)
os.makedirs(result_save_path)
os.makedirs(os.path.join(result_save_path, 'figures'))

sp_deg= np.load('./spatial/degData.npy')
sp_bright = np.load("./spatial/brightData.npy")
sp_mat = np.load("./spatial/matData.npy")
sp_label = np.load("./spatial/labelData.npy")
sp_pred = np.load("./spatial/predictionData.npy")

ch_deg= np.load('./channel/degData.npy')
ch_bright = np.load("./channel/brightData.npy")
ch_mat = np.load("./channel/matData.npy")
ch_label = np.load("./channel/labelData.npy")
ch_pred = np.load("./channel/predictionData.npy")

# print np.shape(sp_pred[0])
# ch_pred=np.concatenate((ch_pred[s*2],ch_pred[s*2+1]),axix=0)
# print np.shape(ch_pred[0])
# exit()




evaluationManager = evt.EvaluationMatrixGenerator()
evaluation_paper =  evt.EvaluationMatrixGenerator()
evaluation_sponge= evt.EvaluationMatrixGenerator()
evaluation_stapler = evt.EvaluationMatrixGenerator()
evaluation_tube = evt.EvaluationMatrixGenerator()


for s in range(len(sp_deg)):

    material = ch_mat[s]
    bright = ch_bright[s]
    degree = ch_deg[s]
    groundTruth = ch_label[s]
    predictionR = (sp_pred[s] + ch_pred[s])/2.0
    #print predictionR[0:10]

    fullpath = '{0}_{1}_{2}'.format(material[0], degree[0], bright[0])
    fullpath = str(fullpath)

    if config.EVAL_SAVE_FIGURES:
        gg.GenerateGraph(predictionR, groundTruth, os.path.join(result_save_path, 'figures'),
                         fullpath + str(s), False)

    evaluationManager.addTable(material, degree, bright, groundTruth, predictionR,'None')
    evaluation_paper.addTable(material, degree, bright, groundTruth, predictionR, 'papercup')
    evaluation_sponge.addTable(material, degree, bright, groundTruth, predictionR, 'sponge')
    evaluation_stapler.addTable(material, degree, bright, groundTruth, predictionR, 'stapler')
    evaluation_tube.addTable(material, degree, bright, groundTruth, predictionR, 'tube')

print result_save_path
evaluationManager.saveTables(result_save_path, 'total')
evaluation_paper.saveTables(result_save_path, 'papercup')
evaluation_sponge.saveTables(result_save_path, 'sponge')
evaluation_stapler.saveTables(result_save_path, 'stapler')
evaluation_tube.saveTables(result_save_path, 'tube')

