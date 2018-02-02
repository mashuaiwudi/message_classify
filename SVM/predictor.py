# -*- coding: utf-8 -*-
'''
# Message Classifier
# Copyright 2017.12 Pulse Analyze Project
#   Author: Ma Shuai
#   All Rights Reserved.
'''

from sklearn.externals import joblib
from preprocessing_data import dimensionality_reduction2
import json
import jieba
import jieba.posseg as pseg
from SVM_Evaluator import Evaluator
from Realtime_Predictor import Predictor
from sklearn.feature_extraction.text import TfidfVectorizer


def cut(doc):

    #list = []
    result = ''
    words = pseg.cut(doc)

    new_doc = ''.join(w.word for w in words if w.flag != 'x')
    words = list(jieba.cut(new_doc))
    for i in words:
        result = result + i + ' '
    #list.append(words)
    #print len(words)
    return result


if '__main__' == __name__:
    test_data = []
    # test_data.append('您好。')
    # test_data.append('红包，充值，您好，理财，收益，x%五万起，三月十号起息，只能柜台购买，数量有限欲购从速！渤海银行祁新星。')
    # test_data.append('经营有九家展览馆的“鳞之家集团”用令人眼花缭乱的绘画和布景让人游客体验到视觉错觉带来的快乐')
    # test_data.append('您好您好您好您好您好您好美女们浦东华润时代广场xxx恩曼琳三八妇女节活动xx年春款x折，xx年秋冬款x.x折，货品有限，现已开始预售，期待各位的光临哦！')

    # content2 = io.mmread('../Data/word_vector.mtx')

    with open('../RawData/train_content.json', 'r') as f:
        content = json.load(f)

    haha = cut('柠檬排毒祛黄亮白按摩霜按摩膏推荐')
    print haha
    test_data.append(haha)

    # test_data += content[0:20]

    # content1 = content[0:1000]
    # for i in content1:
    #    i = i + u'您好'
    #    test_data.append(i)


    vec_tfidf = joblib.load('../model/saved_tfidf2')
    data_tfidf = vec_tfidf.transform(test_data)

    print data_tfidf

    # content = io.mmread('../Data/word_vector.mtx')
    # with open('../Data/train_label.json', 'r') as f:
    #    label = json.load(f)
    # training_data, test_data, training_target, test_target = split_data(content, label)
    # training_data, test_data = dimensionality_reduction(training_data.todense(), test_data.todense(), type='pca')
    # test_data = '圣豪商场开业大酬宾，来店用户送大礼包99元'



    # test_data = vector_word2(test_data)

    test_data = dimensionality_reduction2(data_tfidf.todense(), type='pca')
    clf = joblib.load('model/SVM_linear_estimator.pkl')
    predictor = Predictor(test_data)
    predictor.test_data = test_data
    predictor.sample_predict(clf)
