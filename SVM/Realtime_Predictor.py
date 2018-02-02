# -*- coding: utf-8 -*-
'''
# Message Classifier
# Copyright 2017.12 Pulse Analyze Project
#   Author: Ma Shuai
#   All Rights Reserved.
'''
from sklearn import metrics


class Predictor:
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_target = ''

    def sample_predict(self, clf):
        result = []
        test_result = clf.predict(self.test_data)
        for i in test_result:
            print 'message is'
            print i
            result.append(i)
        return i

        #print metrics.classification_report(self.test_target, test_result)
        #print metrics.confusion_matrix(self.test_target, test_result)

    def new_predict(self, clf):
        test_result = clf.predict(self.test_data)
        with open('result/predict_label.txt', 'wt') as f:
            for i in range(len(test_result)):
                f.writelines(test_result[i])
        self.test_target = test_result
        print 'write over'

