# -*- coding: utf-8 -*-

import json
from scipy import sparse, io
from sklearn.externals import joblib
from SVM_Trainer import TrainerLinear
from SVM_Predictor import Predictor
from preprocessing_data import split_data
from preprocessing_data import dimensionality_reduction
from spam_handle import word2vec_model,data_segment




class Evaluator:
    clf = joblib.load('model/SVM_linear_estimator.pkl')

    def __init__(self, training_data, training_target, test_data, test_target):
        self.trainer = TrainerLinear(training_data, training_target)
        self.predictor = Predictor(test_data, test_target)

    def train(self):
        #self.trainer.learn_best_param()
        self.trainer.train_classifier()
        joblib.dump(self.clf, 'model/Terminal_estimator.pkl')
        Evaluator.clf = joblib.load('model/Terminal_estimator.pkl')

    def cross_validation(self):
        self.trainer.cross_validation()

    def predict(self, type):
        if type == 'sample_data':
            self.predictor.sample_predict(Evaluator.clf)
        elif type == 'new_data':
            self.predictor.new_predict(Evaluator.clf)


if '__main__' == __name__:
    '''
    content = io.mmread('../Data/word_vector.mtx')
    with open('../Data/train_label.json', 'r') as f:
        label = json.load(f)
    training_data, test_data, training_target, test_target = split_data(content, label)
    training_data, test_data = dimensionality_reduction(training_data.todense(), test_data.todense(), type='pca')
    evaluator = Evaluator(training_data, training_target, test_data, test_target)
    #evaluator.train()
    #evaluator.cross_validation()
    evaluator.predict(type='sample_data')
    '''


    content = io.mmread('../Data/word_vector.mtx')
    with open('../Data/train_label.json', 'r') as f:
        label = json.load(f)
    training_data, test_data, training_target, test_target = split_data(content, label)
    #training_data, test_data = dimensionality_reduction(training_data.todense(), test_data.todense(), type='pca')
    evaluator = Evaluator(training_data, training_target, test_data, test_target)


    '''
    #第二种是word2vec
    content = []
    label = []
    lines = []

    # numpy.array(i).reshape(1, -1)

    with open('../rawdata/message.txt') as fr:
        for i in range(10000):
            line = fr.readline()
            lines.append(line.decode('utf-8'))
        num = len(lines)
        for i in range(num):
            message = lines[i].split('\t')
            label.append(int(message[0]))
            content.append(message[1])

    data_segment(content)
    training_data, training_target, test_data, test_target = word2vec_model(content, label)
    print len(training_data[0])
    print len(training_data)
    print len(training_target)
    print len(test_data[0])
    print len(test_data)
    print len(test_target)
    '''




    #evaluator = Evaluator(training_data, training_target, test_data, test_target)

    evaluator.train()
    evaluator.cross_validation()
    evaluator.predict(type='sample_data')
