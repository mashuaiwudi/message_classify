# -*- coding: utf-8 -*-
'''
# Message Classifier
# Copyright 2017.12 Pulse Analyze Project
#   Author: Ma Shuai
#   All Rights Reserved.
'''
import numpy as np
from sklearn import svm
from sklearn import metrics
import json
from scipy import sparse, io
from word_vector import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from scipy import sparse, io
from sklearn.decomposition import PCA
from preprocessing_data import split_data
from preprocessing_data import dimensionality_reduction

# Utility function to move the midpoint of a colormap to be around the values of interest.
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class TrainerLinear:
    def __init__(self, training_data, training_target):
        self.training_data = training_data
        self.training_target = training_target
        self.clf = svm.SVC(C=1, class_weight=None, coef0=0.0,
                           decision_function_shape=None, degree=3, gamma='auto',
                           kernel='linear', max_iter=-1, probability=False,
                           random_state=None, shrinking=True, tol=0.001, verbose=False)

    def learn_best_param(self):
        C_range = np.logspace(-2, 10, 13)
        param_grid = dict(C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        grid.fit(self.training_data, self.training_target)
        self.clf.set_params(C=grid.best_params_['C'])
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))

    def train_classifier(self):
        self.clf.fit(self.training_data, self.training_target)
        joblib.dump(self.clf, 'model/SVM_linear_estimator.pkl')
        training_result = self.clf.predict(self.training_data)
        print metrics.classification_report(self.training_target, training_result)
        #performance_report(self.training_target, training_result)

    def cross_validation(self):
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=20)
        scores = cross_val_score(self.clf, self.training_data, self.training_target, cv=cv, scoring='f1_macro')
        print scores
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



class TrainerRbf:
    def __init__(self, training_data, training_target):
        self.training_data = training_data
        self.training_target = training_target
        self.clf = svm.SVC(C=100, class_weight=None, coef0=0.0,
                           decision_function_shape=None, degree=3, gamma=0.01,
                           kernel='rbf', max_iter=-1, probability=False,
                           random_state=None, shrinking=True, tol=0.001, verbose=False)

    def learn_best_param(self):
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        grid.fit(self.training_data, self.training_target)
        self.clf.set_params(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))
        self.draw_visualization_param_effect(grid, C_range, gamma_range)


    def draw_visualization_param_effect(self, grid, C_range, gamma_range):
        scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                             len(gamma_range))
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest',
                   norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        plt.yticks(np.arange(len(C_range)), C_range)
        plt.title('Validation accuracy')
        plt.savefig('fig/param_effect.png')
        plt.show()


    def train_classifier(self):
        self.clf.fit(self.training_data, self.training_target)
        joblib.dump(self.clf, 'model/SVM_rbf_estimator.pkl')
        training_result = self.clf.predict(self.training_data)
        print metrics.classification_report(self.training_target, training_result)


    def cross_validation(self):
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=20)
        scores = cross_val_score(self.clf, self.training_data, self.training_target, cv=cv, scoring='f1_macro')
        print scores
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def performance_report(target, result):
    confusion = metrics.confusion_matrix(target, result)
    print 'confusion matrix'
    print confusion

    TP = int(confusion[0, 0])
    FN = int(confusion[0, 1])
    FP = int(confusion[1, 0])
    TN = int(confusion[1, 1])

    # 下面是我自己注释掉的，最后要取消掉注释
    # 下面全面的衡量这个分类器的效果
    Accuracys = float(TP + TN) / (TP + FP + TN + FN)

    Precisions = float(TP) / (TP + FP)

    Recalls = float(TP) / (TP + FN)  # recall

    f_value = 2 * Recalls * Precisions / (Recalls + Precisions)

    # print("TP:" + str(TP))
    # print("TN:" + str(TN))
    # print("FP:" + str(FP))
    # print("FN:" + str(FN))
    print("Recalls是：%s" % str(Recalls))
    print("Precisions：%s" % str(Precisions))
    print("Accuracys：%s" % str(Accuracys))
    print("f_value：%s" % str(f_value))



def SVM_train(train_data, train_target):
    clf = svm.SVC(kernel='linear', class_weight='balanced', C =100, gamma = 0.01)
    clf.fit(train_data, train_target)
    expected = train_target
    predicted = clf.predict(train_data)
    # summarize the fit of the model
    print metrics.classification_report(expected, predicted)
    print metrics.confusion_matrix(expected, predicted)


def feature_selection(data, data_target, feature_names):
    clf = svm.SVC(class_weight='balanced', C =2)
    clf.fit(data, data_target)




if '__main__' == __name__:
    content = io.mmread('../Data/word_vector.mtx')
    with open('../Data/train_label.json', 'r') as f:
        label = json.load(f)
    training_data, test_data, training_target, test_target = split_data(content, label)
    training_data, test_data = dimensionality_reduction(training_data.todense(), test_data.todense(), type='pca')

    #Trainer = TrainerLinear(training_data, training_target)
    Trainer = TrainerRbf(training_data, training_target)
    #Trainer.learn_best_param()
    Trainer.train_classifier()

    #Trainer.cross_validation()
    print 'finished'


    #Trainer2 = TrainerRbf(training_data, training_target)
    #Trainer2.learn_best_param()
    #Trainer2.train_classifier()
    #Trainer2.cross_validation()


