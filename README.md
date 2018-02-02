# message_classify
对垃圾短信进行分类
代码分为如下几个模块
## 一、RawData
### 该文件夹中存储了原始的短信数据

## 二、Data
### 该文件夹中存储了处理后的数据

## 三、model
### 该文件夹中存储了fit好的pca模型和tfidf模型，目的是对新来的短信数据可以快速进行处理

## 四、data_parse
### 1.load_data.py
#### 该代码的作用是加载原始短信数据，然后将数据拆分为内容和标签，分别存储，便于后续操作
### 2.preprocessing_data.py
#### 该代码中包括对数据进行训练集和测试集的切分、标准化、降维三个功能
### 3.word_vector.py
#### 该代码的作用是将短信数据进行处理，计算tfidf特征
### 4.spam_handler.py
#### 该代码的作用是使用word2vec计算word vector特征

## 五、classifier_sklearn
### 该文件夹下是通过调用sklearn中的包对模型进行训练
### 1.model
#### 该文件夹下存储各个分类器训练出的模型
### 2.Train.py
#### 该代码的作用是对各个分类器模型进行训练，并可以进行交叉验证。对每一个分类器，写了一个class，每一个class中包含train_classifier、cross_validation两个函数
### 3.Predictor.py
#### 该代码是用训练好的分类器对测试数据进行分类，并进行recall、precision、f1等指标的计算
### 4.Evaluator.py
#### 该代码的作用是对分类器进行评估，包括了Train.py里的训练过程，以及用测试集进行测试的过程。

## 六、SVM
### 1.fig
#### 该文件夹中存储对SVM中参数C进行学习的过程中的最好的参数的图
### 2.model
#### 该文件夹中存储了线性SVM和RBF核的SVM的训练模型
### 3.SVM_Trainer.py
#### 该代码的作用是对SVM分类器进行训练，并交叉验证
### 4.SVM_Predictor.py
#### 该代码的作用是用训练好的SVM模型对测试数据进行预测
### 5.SVM_Evaluator.py
#### 该代码的作用是结合上述两个代码的功能，先进行训练然后再进行测试。
### 6.Realtime_Predictor.py
#### 该代码的作用是对一个新来的数据进行预测然后返回分类结果
### 7.predictor.py
#### 该代码的作用是手动的输入一条短信，然后查看分类结果

## 七、SVM_mine
### 1.data
#### 该文件夹中存储了word2vector所需要的切分结果
### 2.model
#### 该文件夹中存储了svm模型和word2vec模型
### 3.svm_interface.py
#### 该代码提供了svm的一些接口
### 4.svm_by_ms.py
#### 该代码是自己实现的SVM的算法
### 5.predictor.py
#### 该代码是对整个自己写的SVM进行测试

## 八、KNN_mine
### 1.knn.py
#### 该代码是自己实现的KNN算法

## 九、MyApplication
### 这是写的Android程序，功能为拦截手机接收的短信然后发送给server进行分类，返回分类结果等。

## 十、server.py
### 该代码是通过socket简单实现了一个server，实现了接收一条短信，调用训练好的模型对其进行分类。

## 十一、client.py
### 该代码是通过python实现的一个简单的客户端，功能是向server发送一条短信，然后接收分类结果


## 代码运行顺序（以classifier_sklearn中的为例）
### 1.先运行load_data.py，将原始的短信数据加载进来并进行content和label的切分，这里可以修改读取的短信的条数
### 2.然后运行word_vector.py，可以计算tfidf特征，并存储为稀疏矩阵的形式
### 3.然后运行Train.py模型训练，这时候可以选择加载tfidf，也可以选择加载原始数据计算word2vec特征，还可以选择使用哪个分类器。
### 4.然后运行Evaluator.py，就可以用训练好的模型对测试集进行测试，也可以不运行第三步中的Train.py，只运行Evaluator.py，这个代码中包含了训练和测试。
### 5.Predictor.py，这个是一个实时短信分类的测试代码，可以自己在代码里修改输入的短信，会得到分类结果
### 6.运行server.py，会开启一个socket连接，然后运行client.py或者安卓APP，即可以实现“云端”的实时分类。
