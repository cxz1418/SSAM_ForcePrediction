# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import numpy as np

def GenerateGraph(_prediction, _groundTruth,_savePath,_fileName,_willShow):
    #---input---
    #pred : array1[]

    #---output---
    #1.graph image 2.evaluation_config 3.vectorValues

    #groundTruth ; array2[]
    #This function will print the following output
    #1.Comparison Graph
    #2.output.textFile (SNR,MSE)
    rmse = np.sqrt(((np.array(_prediction) - np.array(_groundTruth)) ** 2).mean(axis=0))
    snr =0

    fig, ax = plt.subplots(1)
    fig.autofmt_xdate()


    ylim = max(np.max(_prediction), np.max(_groundTruth)) + 0.1
    ybot = min(np.min(_prediction), np.min( _groundTruth)) - 0.1
    ax.set_ylim([ybot, ylim])
    ax.set_xlim([0, 200])

    plt.xlabel('Frame')
    plt.ylabel('Force(N)')
    plot_test, = ax.plot(_groundTruth, color='lime', linestyle='--', label='Ground Truth',linewidth=1.0)
    plot_predicted, = ax.plot(_prediction, color='red', linestyle='-', label='Proposed Method',linewidth=1.0)
    plt.legend(handles=[plot_predicted, plot_test], loc='upper right')


    if(_willShow):
        plt.show()


    #############Save Result###############

    #write Comparison Graph
    fig.savefig(_savePath + '/'+_fileName+'_Graph.png')



    #write Vector txtFile
    fin_vec = open(_savePath+'/'+_fileName+'_Vector.txt','wb')
    np.save(fin_vec,_prediction)
    np.save(fin_vec,_groundTruth)
    fin_vec.close()

    plt.close()


if __name__ =='__main___':
    test_array =[]
    pred_array=[]
    for k in range(50):
        test_array.append((k%4))
        pred_array.append((k %10))

    GenerateGraph(pred_array,test_array,'./ResultFold','test')

