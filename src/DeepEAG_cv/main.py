#!/usr/bin/env python
#_*_coding:utf-8_*_
import argparse
import re,csv,math
from codes import *
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  
        SaveList.append(row)
    return
def ReadMyTsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName),delimiter = '\t')
    for row in csv_reader:  
        SaveList.append(row)
    return
def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return
def MyConfusionMatrix(y_real,y_predict):
    from sklearn.metrics import confusion_matrix
    CM = confusion_matrix(y_real, y_predict)
    print(CM)
    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    print('TN:%d, FP:%d, FN:%d, TP:%d' % (TN, FP, FN, TP))
    Acc = (TN + TP) / (TN + TP + FN + FP)
    Sen = TP / (TP + FN)
    Spec = TN / (TN + FP)
    Prec = TP / (TP + FP)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    Rec = TP/(TP+FN)
    F1=(2*Prec*Rec)  / (Prec+Rec)
    
    # print('Acc:', round(Acc, 4))
    # print('Sen:', round(Sen, 4))
    # print('Spec:', round(Spec, 4))
    # print('Prec:', round(Prec, 4))
    # print('MCC:', round(MCC, 4))
    Result = []
    Result.append(round(Acc, 8))
    Result.append(round(Sen, 8))
    Result.append(round(Spec, 8))
    Result.append(round(Prec, 8))
    Result.append(round(MCC, 8))
    Result.append(round(Rec, 8))
    Result.append(round(F1, 8))
    return Result
def MyRealAndPredictionProb(Real,prediction):
    RealAndPredictionProb = []
    counter = 0
    while counter < len(prediction):
        pair = []
        pair.append(Real[counter])
        pair.append(prediction[counter][0])
        pair.append(prediction[counter][1])
        RealAndPredictionProb.append(pair)
        counter = counter + 1
    return RealAndPredictionProb
def MyRealAndPrediction(Real,prediction):
    RealAndPrediction = []
    counter = 0
    while counter < len(prediction):
        pair = []
        pair.append(Real[counter])
        pair.append(prediction[counter])
        RealAndPrediction.append(pair)
        counter = counter + 1
    return RealAndPrediction
if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="it's usage tip.", description="Training and validating model by 5-fold cv")
    parser.add_argument("--feature_path", required=True, help="feature file path")
    parser.add_argument("--label_path", required=True, help="label file path")
    args = parser.parse_args()
    feature_path=args.feature_path
    label_path=args.label_path
    print("---please do not change the feature file name && Read features---\n")
    AAC_Feature = []
    ReadMyTsv(AAC_Feature, feature_path + "/AAC_encoding.tsv")
    AAC_Feature = AAC_Feature[1:][:] # Delete table header
    
    PAAC_Feature = []
    ReadMyTsv(PAAC_Feature,feature_path+"/PAAC_encoding.tsv")
    PAAC_Feature = PAAC_Feature[1:][:] # Delete table header
    
    OneHot_Feature = []
    ReadMyTsv(OneHot_Feature, feature_path + "/OHE_encoding.tsv")
    OneHot_Feature = OneHot_Feature[1:][:] # Delete table header
    
    PSSM_Com_Feature = []
    ReadMyTsv(PSSM_Com_Feature, feature_path + "/PSSMC_encoding.tsv")
    
    Label=[]
    ReadMyCsv(Label, label_path + "/Label.csv")#
    Label = np.array(Label)
    name = Label[:, 0]
    label = Label[:, 1]
    print("---start cross validation analysis---\n")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    k=0
    for train_index, test_index in skf.split(name, label):
        train_name, train_label = name[train_index], label[train_index]
        test_name, test_label = name[test_index], label[test_index]
        print("--HyperVR-ARG---\n")
        final_blend_train = []
        fianl_blend_test_mean = []
        print("---deep learning && onehot features---\n")
        final_blend_train1, fianl_blend_test_mean1 = DL_Bitscore.DL_Bitscore(train_name, train_label, test_name,test_label,Bit_Score_Feature)
        print("---classifier && PSSMC features---\n")
        final_blend_train2, fianl_blend_test_mean2 = ET_Pssmc.ET_Pssmc(train_name, train_label, test_name,test_label, PSSM_Com_Feature)
        
        print("---Stacking model && All features---\n")
        for i in range(len(final_blend_train1)):
            pair = []
            pair.append(final_blend_train1[i][0])
            pair.append(final_blend_train1[i][1])
            pair.append(final_blend_train2[i][1])
            pair.append(final_blend_train3[i][1])
            pair.append(final_blend_train4[i][1])
            final_blend_train.append(pair)
        for i in range(len(fianl_blend_test_mean1)):
            pair = []
            pair.append(fianl_blend_test_mean1[i][0])
            pair.append(fianl_blend_test_mean1[i][1])
            pair.append(fianl_blend_test_mean2[i][1])
            pair.append(fianl_blend_test_mean3[i][1])
            pair.append(fianl_blend_test_mean4[i][1])
            fianl_blend_test_mean.append(pair)
        final_blend_train = np.array(final_blend_train, dtype=float)
        fianl_blend_test_mean = np.array(fianl_blend_test_mean, dtype=float)
        final_blend_train_data, final_blend_train_label = final_blend_train[:, 1:], final_blend_train[:, 0]
        final_blend_test_data, final_blend_test_label = fianl_blend_test_mean[:, 1:], fianl_blend_test_mean[:, 0]
        clfStack = Xgboost()
        clfStack.fit(final_blend_train_data, final_blend_train_label)
        y_score0 = clfStack.predict(final_blend_test_data)
        y_score1 = clfStack.predict_proba(final_blend_test_data)
        RealAndPrediction = MyRealAndPrediction(final_blend_test_label, y_score0) #Real and predicted labels for the first step ARG prediction
        RealAndPredictionProb = MyRealAndPredictionProb(final_blend_test_label, y_score1) #Real and predicted probabilities for the first step ARG prediction
        StorFile(RealAndPrediction,str(k)+"RealAndPredictionForARG.csv") # k fold
        StorFile(RealAndPredictionProb, str(k)+"RealAndPredictionProbForARG.csv")# k fold
        clfStackResult = MyConfusionMatrix(final_blend_test_label, y_score0)
        print("Stacking model (HyperVR-ARG) Test accuracy: precision: recall: f1:", clfStackResult[0], clfStackResult[3],
              clfStackResult[5], clfStackResult[6])
        print("--HyperVR-VF---\n")
        test_name2 = []
        test_label2 = []

        for i in range(len(RealAndPrediction)):# Predicting the remaining genes that are not ARGs
            if RealAndPrediction[i][1] == 0.0:
                test_name2.append(name[test_index[i]])
                test_label2.append(label[test_index[i]])
        test_name = np.array(test_name2)
        test_label = np.array(test_label2)
        final_blend_train = []
        fianl_blend_test_mean = []
        print("---random forest && AAC features---\n")
        final_blend_train1, fianl_blend_test_mean1 = RF_AAC.RF_AAC(train_name, train_label, test_name, test_label, AAC_Feature)
        print("---xgboost && AAC features---\n")
        final_blend_train2, fianl_blend_test_mean2 = Xgboost_AAC.Xgboost_AAC(train_name, train_label, test_name, test_label, AAC_Feature)
        print("---svm && AAC features---\n")
        final_blend_train3, fianl_blend_test_mean3 = ET_AAC.SVM_AAC(train_name, train_label, test_name, test_label, AAC_Feature)
        print("---adaboost && AAC features---\n")
        final_blend_train5, fianl_blend_test_mean5 = AB_AAC.AB_AAC(train_name, train_label, test_name, test_label, AAC_Feature)

        print("---random forest && PAAC features---\n")
        final_blend_train1, fianl_blend_test_mean1 = RF_AAC.RF_PAAC(train_name, train_label, test_name, test_label, AAC_Feature)
        print("---xgboost && AAC features---\n")
        final_blend_train2, fianl_blend_test_mean2 = Xgboost_AAC.Xgboost_PAAC(train_name, train_label, test_name, test_label, AAC_Feature)
        print("---svm && AAC features---\n")
        final_blend_train3, fianl_blend_test_mean3 = ET_AAC.SVM_PAAC(train_name, train_label, test_name, test_label, AAC_Feature)
        print("---adaboost && AAC features---\n")
        final_blend_train5, fianl_blend_test_mean5 = AB_AAC.AB_PAAC(train_name, train_label, test_name, test_label, AAC_Feature)

        print("---random forest && PSSM features---\n")
        final_blend_train1, fianl_blend_test_mean1 = RF_AAC.RF_PSSM(train_name, train_label, test_name, test_label, AAC_Feature)
        print("---xgboost && PSSM features---\n")
        final_blend_train2, fianl_blend_test_mean2 = Xgboost_AAC.Xgboost_PSSM(train_name, train_label, test_name, test_label, AAC_Feature)
        print("---svm && PSSM features---\n")
        final_blend_train3, fianl_blend_test_mean3 = ET_AAC.SVM_PSSM(train_name, train_label, test_name, test_label, AAC_Feature)
        print("---adaboost && PSSM features---\n")
        final_blend_train5, fianl_blend_test_mean5 = AB_AAC.AB_PSSM(train_name, train_label, test_name, test_label, AAC_Feature)


        print("---deep learning && one hot features---\n")
        final_blend_train41, fianl_blend_test_mean41 = DL_OHE.DL_OHE(train_name, train_label, test_name,test_label, OneHot_Feature)

        print("---Stacking model && All features---\n")
        for i in range(len(final_blend_train1)):
            pair = []
            pair.append(final_blend_train1[i][0])
            pair.append(final_blend_train1[i][1])
            pair.append(final_blend_train2[i][1])
            pair.append(final_blend_train3[i][1])
            pair.append(final_blend_train4[i][1])
            pair.append(final_blend_train5[i][1])
            pair.append(final_blend_train6[i][1])
            pair.append(final_blend_train7[i][1])
            pair.append(final_blend_train8[i][1])
            pair.append(final_blend_train9[i][1])
            pair.append(final_blend_train10[i][1])
            pair.append(final_blend_train11[i][1])
            pair.append(final_blend_train12[i][1])
            pair.append(final_blend_train13[i][1])
            final_blend_train.append(pair)
        for i in range(len(fianl_blend_test_mean1)):
            pair = []
            pair.append(fianl_blend_test_mean1[i][0])
            pair.append(fianl_blend_test_mean1[i][1])
            pair.append(fianl_blend_test_mean2[i][1])
            pair.append(fianl_blend_test_mean3[i][1])
            pair.append(fianl_blend_test_mean4[i][1])
            pair.append(fianl_blend_test_mean5[i][1])
            pair.append(fianl_blend_test_mean6[i][1])
            pair.append(fianl_blend_test_mean7[i][1])
            pair.append(fianl_blend_test_mean8[i][1])
            pair.append(fianl_blend_test_mean9[i][1])
            pair.append(fianl_blend_test_mean10[i][1])
            pair.append(fianl_blend_test_mean11[i][1])
            pair.append(fianl_blend_test_mean12[i][1])
            pair.append(fianl_blend_test_mean13[i][1])
            fianl_blend_test_mean.append(pair)
        final_blend_train = np.array(final_blend_train, dtype=float)
        fianl_blend_test_mean = np.array(fianl_blend_test_mean, dtype=float)
        final_blend_train_data, final_blend_train_label = final_blend_train[:, 1:], final_blend_train[:, 0]
        final_blend_test_data, final_blend_test_label = fianl_blend_test_mean[:, 1:], fianl_blend_test_mean[:, 0]
        clfStack = Xgboost()
        clfStack.fit(final_blend_train_data, final_blend_train_label)
        y_score0 = clfStack.predict(final_blend_test_data)  #
        y_score1 = clfStack.predict_proba(final_blend_test_data)  #
        clfStackResult = MyConfusionMatrix(final_blend_test_label, y_score0)
        RealAndPrediction = MyRealAndPrediction(final_blend_test_label, y_score0)#Real and predicted labels for the Second step VF prediction
        RealAndPredictionProb = MyRealAndPredictionProb(final_blend_test_label, y_score1)#Real and predicted probabilities for the Second step VF prediction
        StorFile(RealAndPrediction, str(k) + "RealAndPredictionForVF.csv")  # k fold
        StorFile(RealAndPredictionProb, str(k) + "RealAndPredictionProbForVF.csv")  # k fold
        print("Stacking model (HyperVR-VF) Test accuracy: precision: recall: f1:", clfStackResult[0], clfStackResult[3],
              clfStackResult[5], clfStackResult[6])
        k+=1
