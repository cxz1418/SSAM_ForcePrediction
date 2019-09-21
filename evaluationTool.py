# -*- coding: utf-8 -*-
import numpy as np
import csv


def GetMSE(_predicted, _targets):
    return np.mean((_targets - _predicted)**2)

def GetRMSE(_predicted, _targets):
    return (np.sqrt(np.mean((_targets-_predicted)**2)))

def GetMAE(_predicted, _targets):
    return(np.mean(np.absolute(_targets - _predicted)))


class EvaluationMatrixGenerator:
    def __init__(self):
        self.sampleNum=0.0
        self.materialEvalTable_sampleNum = np.zeros(4)
        self.envEvalTable_sampleNum = np.zeros((3, 4))

        self.materialEvalTable_MAE = np.zeros(4)
        self.envEvalTable_MAE = np.zeros((3, 4))

        self.materialEvalTable_MSE = np.zeros(4)
        self.envEvalTable_MSE = np.zeros((3, 4))


    def addTable(self,_materialArr,_degreeArr,_brightArr,_groundTruthArr,_predictionArr,_matType):
        for i in range(len(_predictionArr)):

            _degree = _degreeArr[i]
            _bright = _brightArr[i]
            _material = _materialArr[i]
            _groundTruth = _groundTruthArr[i]
            _prediction = _predictionArr[i]


            if(_matType!='None'):
                if(_material!=_matType):
                    continue


            if (_degree == '0'):
                di = 0
            elif (_degree == '10'):
                di = 1
            elif (_degree == '20'):
                di = 2
            elif (_degree == '30'):
                di = 3

            else:
                print 'Get Wrong enviroment'
                print _degree

            if (_bright == '50'):
                bi = 0
            elif(_bright=='75'):
                bi = 1
            elif (_bright == '100'):
                bi = 2
            else:
                print 'Get Wrong enviroment'
                print _bright


            if(_material=='papercup'):
                mi=0
            elif (_material == 'sponge'):
                mi = 1
            elif (_material == 'stapler'):
                mi = 2
            elif (_material == 'tube'):
                mi = 3
            else:
                print 'Get Wrong enviroment'
                print _material

            mae =GetMAE(_prediction,_groundTruth)

            mse = GetMSE(_prediction,_groundTruth)


            self.materialEvalTable_sampleNum[mi]+=1
            self.envEvalTable_sampleNum[bi][di]+=1

            self.envEvalTable_MAE[bi][di]+=mae
            self.envEvalTable_MSE[bi][di] += mse


            self.materialEvalTable_MAE[mi]+=mae
            self.materialEvalTable_MSE[mi] += mse


    def GetEnvEvalTable(self):
        return self.envEvalTable_MAE  ,self.envEvalTable_MSE


    def GetMatEvalTable(self):
        return self.materialEvalTable_MAE  ,self.materialEvalTable_MSE


    def addMeanResult(self,arr, _c, _r):
        arr = np.concatenate((arr, np.reshape(np.mean(arr, axis=0), (1, _r))), axis=0)
        arr = np.concatenate((arr, np.reshape(np.mean(arr, axis=1), (_c+1, 1))), axis=1)
        return arr



    def writeEnvTable_csv(self,_csv_wr,_name,_table):
        envRow_1 = np.array([_name, '0_deg', '10_deg', '20_deg', '30_deg', 'total'])
        envRow_2 = np.array(['50_bright'])
        envRow_2 = np.concatenate((envRow_2, _table[0]), axis=0)
        envRow_3 = np.array(['75_bright'])
        envRow_3 = np.concatenate((envRow_3, _table[1]), axis=0)
        envRow_4 = np.array(['100_bright'])
        envRow_4 = np.concatenate((envRow_4, _table[2]), axis=0)
        envRow_5 = np.array(['avg'])
        envRow_5 = np.concatenate((envRow_5, _table[3]), axis=0)

        _csv_wr.writerow(envRow_1)
        _csv_wr.writerow(envRow_2)
        _csv_wr.writerow(envRow_3)
        _csv_wr.writerow(envRow_4)
        _csv_wr.writerow(envRow_5)
        _csv_wr.writerow([' '])

    def divideBySampleNum(self,arr1,arr2):
        return np.divide(arr1,arr2,out=np.zeros_like(arr1),where=arr2!=0)

    def saveTables(self,_savePath,_name):
        fout = open(_savePath+'/EvalTable_%s.csv'%(_name),'wb')

        MatTable_mae ,MatTable_mse = self.GetMatEvalTable()
        EnvTable_mae ,EnvTable_mse = self.GetEnvEvalTable()
        MatTable_mae= self.divideBySampleNum(MatTable_mae,self.materialEvalTable_sampleNum)
        MatTable_mse=self.divideBySampleNum(MatTable_mse,self.materialEvalTable_sampleNum)


        EnvTable_mae=self.divideBySampleNum(EnvTable_mae,self.envEvalTable_sampleNum)
        EnvTable_mae = self.addMeanResult(EnvTable_mae,3,4)
        EnvTable_mse=self.divideBySampleNum(EnvTable_mse,self.envEvalTable_sampleNum)
        EnvTable_mse = self.addMeanResult(EnvTable_mse, 3, 4)
        EnvTable_rmse = np.sqrt(EnvTable_mse)

        wr = csv.writer(fout)


        matRow_1 =np.array([' ','papercup','sponge','stapler','tube','avg'])
        matRow_2 = np.array(['RMSE'])
        matRow_2 = np.concatenate((matRow_2,np.sqrt(MatTable_mse),np.sqrt([np.mean(MatTable_mse)])),axis=0)
        matRow_3 = np.array(['MAE'])
        matRow_3 = np.concatenate((matRow_3, MatTable_mae,[np.mean(MatTable_mae)]), axis=0)
        matRow_4 = np.array(['MSE'])
        matRow_4 = np.concatenate((matRow_4, MatTable_mse, [np.mean(MatTable_mse)]), axis=0)

        wr.writerow(matRow_1)
        wr.writerow(matRow_2)
        wr.writerow(matRow_3)
        wr.writerow(matRow_4)
        wr.writerow([' '])


        #draw_envTable(_name,envTable)

        self.writeEnvTable_csv(wr,'RMSE',EnvTable_rmse)
        self.writeEnvTable_csv(wr, 'MAE', EnvTable_mae)
        self.writeEnvTable_csv(wr,'MSE',EnvTable_mse)

        fout.close()

