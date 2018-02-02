# -*- coding: utf-8 -*-
'''
# Message Classifier
# Copyright 2017.12 Pulse Analyze Project
#   Author: Ma Shuai
#   All Rights Reserved.
'''
from spam_handle import data_segment,word2vec_model,model_save,report
from svm_by_ms import SVM,predict_result
if __name__ =="__main__":
    '''
    label_data = pd.read_table('data/labeled.txt')
    content_data = label_data[["Content"]]
    label_data = label_data[["Label"]]
    label_data = label_data.values
    label_data.shape = (1,len(label_data))
    label_data = label_data[0]
    '''
    content = []
    label = []
    lines = []

    #numpy.array(i).reshape(1, -1)

    with open('../data/message.txt') as fr:
        for i in range(100):
            line = fr.readline()
            lines.append(line.decode('utf-8'))
        num = len(lines)
        for i in range(num):
            message = lines[i].split('\t')
            label.append(int(message[0]))
            content.append(message[1])

    data_segment(content)

    for i in range(len(label)):
        if(label[i] == 0):
            label[i]= -1
    print label[:10], content[:10]
    X_train,Y_train,X_test,Y_test = word2vec_model(content,label)
    #测试自己实现的svm
    C = 0.6
    #toler = 0.001
    toler = 0.1
    maxIter = 10
    #maxIter = 100
    test = SVM(X_train, Y_train, C, toler, 1, maxIter);
    print "step 1: training..."
    alphs_result, x_result, y_result, b = test.train_svm()
    model_save(alphs_result, x_result, y_result, b )
    print "step 2: testing..."
    Y_predict = predict_result(X_test, alphs_result, x_result, y_result, b)
    print "step 3: show the result..."
    for index, item in enumerate(Y_test):
        Y_test[index] = int(item)
    report(Y_test, Y_predict)
    print(Y_predict)