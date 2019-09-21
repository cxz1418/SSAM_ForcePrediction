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


def GetBaseLine():
    foldPath='./data/0/'
    sp_label = np.load(foldPath + "spatial/labelData.npy")
    gt_array =[]
    for s in range(len(sp_label)):
        for z in range(len(sp_label[s])):
            gt_array.append(sp_label[s][z])

    return np.array(gt_array)

def GetThresholdResult(baseline,pred):

    absm = np.absolute(baseline-pred)
    low=[]
    high=[]



    for s in range(len(baseline)):
        if(baseline[s]<2):
            low.append(absm[s])
        else:
            high.append(absm[s])

    print (np.average(low)), (np.average(high))


def GetPredictionValue(index):


    foldPath ='./data/'+str(index)+'/'

    sp_deg= np.load(foldPath+'spatial/degData.npy')
    sp_bright = np.load(foldPath+"spatial/brightData.npy")
    sp_mat = np.load(foldPath+"spatial/matData.npy")
    sp_label = np.load(foldPath+"spatial/labelData.npy")
    sp_pred = np.load(foldPath+"spatial/predictionData.npy")

    ch_deg= np.load('channel/degData.npy')
    ch_bright = np.load("channel/brightData.npy")
    ch_mat = np.load("channel/matData.npy")
    ch_label = np.load("channel/labelData.npy")
    ch_pred = np.load("channel/predictionData.npy")



    prediction_array =[]

    for s in range(len(sp_deg)):
        for z in range(len(sp_deg[s])):
            prediction_array.append((sp_pred[s][z]+ch_pred[s][z])/2.0)


    return np.array(prediction_array)

baseline = GetBaseLine()
frame1 =GetPredictionValue(0)
frame2 = GetPredictionValue(1)
frame3 = GetPredictionValue(2)


GetThresholdResult(baseline,frame1)
GetThresholdResult(baseline,frame2)
GetThresholdResult(baseline,frame3)

exit()
print np.average(baseline-frame1)
print np.average(baseline-frame2)
print np.average(baseline-frame3)
print np.shape(baseline-frame1)