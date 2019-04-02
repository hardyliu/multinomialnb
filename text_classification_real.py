# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:21:01 2019

@author: hardyliu
"""

import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


# 加载停用词
with open('./text classification/stop/stopword.txt', 'rb') as f:
    STOP_WORDS = [line.strip() for line in f.readlines()]

#加载训练集和测试集
def load_data(base_path):
    """
    :param base_path: 基础路径
    :return: 分词列表，标签列表
    """
    documents = []
    labels = []

    for root, dirs, files in os.walk(base_path): # 循环所有文件并进行分词打标
        label = root.split('\\')[-1] # 因为windows上路径符号自动转成\了，所以要转义下
        for file in files:                
            labels.append(label)
            filename = os.path.join(root, file)
            with open(filename, 'rb') as f: # 因为字符集问题因此直接用二进制方式读取
                content = f.read()
                word_list = list(jieba.cut(content))
                documents.append(' '.join(word_list))
    return documents, labels


def train_fun(train_document, train_labels, test_document, test_labels):
    """
    构造模型并计算测试集准确率
    :param train_document: 训练集数据
    :param train_labels: 训练集标签
    :param test_document: 测试集数据
    :param test_labels: 测试集标签
    :return: 测试集准确率
    """
    # 计算矩阵
    tt = TfidfVectorizer(stop_words=STOP_WORDS, max_df=0.5)
    tf = tt.fit_transform(train_document)
    # 训练模型
    clf = MultinomialNB(alpha=0.001).fit(tf, train_labels)
    # 模型预测
    test_tf = TfidfVectorizer(stop_words=STOP_WORDS, max_df=0.5, vocabulary=tt.vocabulary_)
    test_features = test_tf.fit_transform(test_document)
    predicted_labels = clf.predict(test_features)
    # 获取结果
    score = metrics.accuracy_score(test_labels, predicted_labels)
    return score


# text classification与代码同目录下
train_documents, train_labels = load_data('./text classification/train')
test_documents, test_labels = load_data('./text classification/test')
score = train_fun(train_documents, train_labels, test_documents, test_labels)
print("准确率：",score)