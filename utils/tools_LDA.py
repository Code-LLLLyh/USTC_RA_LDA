# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 10:09:37 2022

@author: 92853
"""
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import time 
import pandas as pd
import numpy as np

class tools_LDA():
    '''
    第一部分：获得LDA模型 decideKeyNum，resultLDA
    第二部分：将已经得到的模型进行应用，获得对应数据的主题、困惑度等
    '''
    def __init__(self):
        self.workContext='应用工具包'
        
    def decideKeyNum(self,dataPreDeal,path):
        '''
        确定最优主题的数量
        :param dataPreDeal 结构变量 预处理过的数据
        :param path str  如：'data/Model/'
        :return topicNum int 最优的主题数
        '''
        plexs = []          #各个主题数对应的困惑度
        scores = []         #极大似然值
        maxTopicsNum = 15
        for i in range(15,maxTopicsNum+1):
            start=time.time()
            LDAModel = LatentDirichletAllocation(n_components=i,
                                            max_iter=50,
                                            learning_method='online',
                                            learning_offset=50,random_state=0)
            LDAModel.fit(dataPreDeal)
            plexs.append(LDAModel.perplexity(dataPreDeal))
            scores.append(LDAModel.score(dataPreDeal))
            LDAModel=np.array([LDAModel])
            np.save(path+'LDAModel'+str(i)+'.npy',LDAModel)
            end=time.time()
            print('已检查主题数为'+str(i)+'时的困惑度，用时'+str(round(end-start,2))+'秒')
            
        return plexs,scores
        
    def resultLDA(self,textContext,keyNum,printNum,path,isTopic=False):
        '''
        使用LDA分析数据提取数据
        :param textContext DataFrame 使用的数据
        :param keyNum int 特征词数量
        :param printNum int 每个主题输出的词语
        :param isTopic logic 是否要获得最优的主题数
        :param path str 模型储存的位置
        '''
        ##首先设计模型
        #向量化模型
        vectorModel = CountVectorizer(strip_accents = 'unicode',
                                max_features=keyNum,            #参数确定每个主题提取的特征个数
                                stop_words='english',
                                max_df = 0.5,                   #特征值入选的上阈值
                                min_df = 10)                    #特征值入选的下阈值
        
        #数据处理
        #fit_transform是函数fit和transform的集合包括求数据的均值、最大值最小值、归一化等操作
        #详细说明可参考:https://blog.csdn.net/weixin_38278334/article/details/82971752
        dataPreDeal=vectorModel.fit_transform(textContext.TextCut)
        print('已构建完成规则化数据')
        
        #获取最优的主题数
        if isTopic==False:
            topicNum=30
            plexs=['no plexs']
            scores=['no scores'];
        else:
            print('开始尝试寻找最优主题数')
            plexs,scores=self.decideKeyNum(dataPreDeal,path) #得到困惑都和极大似然函数
            topicNum=plexs.index(min(plexs))+1 #这里有错误
        
        #LDA模型设计
        start=time.time()
        print('训练过程开始')
        #参数详解可参考:https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
        LDAModel = LatentDirichletAllocation(n_components=topicNum,
                                max_iter=50,
                                learning_method='online',
                                learning_offset=50,
#                               doc_topic_prior=0.1,
#                               topic_word_prior=0.01,
                                random_state=0)
        LDAModel.fit(dataPreDeal)
        np.save(path+'LDAModel'+str(topicNum)+'.npy',LDAModel)
        end=time.time()
        print('训练结束，训练过程耗时:'+str(round(end-start,2))+'秒')
    
    def vectorData(self,keyNum,data):
        '''
        :param keyNum int 关键词的数量
        :param data dataframe 原始数据
        :return vectorModel VectorModule 向量词模型
        :return data_Stand array 词数据
        '''
        vectorModel = CountVectorizer(strip_accents = 'unicode',
                    max_features=keyNum,            #参数确定每个主题提取的特征个数
                    stop_words='english',
                    max_df = 0.5,                   #特征值入选的上阈值
                    min_df = 10)                    #特征值入选的下阈值
        data_Stand=vectorModel.fit_transform(data.TextCut)
        print('已构建完成规则化数据')
        
        return vectorModel,data_Stand
    
    def getTopicsWithModel(self,printNum,vectorModel,LDAModel):
        '''
        从已保存的模型中获取主题词
        :param printNum int 输出的关键词数量
        :param vectorModel vectorModule 向量词模型
        :param LDAModel ldaModule 模型储存在了数组中
        :return topicWordList list 储存各主体关键词的数量
        '''
        #输出主题已经对应的关键词汇
        dataName = vectorModel.get_feature_names()
        topicWordsList=[]
        for topic_idx, topic in enumerate(LDAModel.components_):
            topicWords = " ".join([dataName[i] for i in topic.argsort()[:-printNum - 1:-1]])
            topicWordsList.append(topicWords) 
        
        return topicWordsList
    
    
if __name__ == '__main__':
    
    tool=tools_LDA()
    #读取需要使用的数据
    textContext=[]
    for j in range(0,15):
        startSpecial=time.time()
        #这里使用了绝对路径注意修改
        data=pd.read_excel('E:/2022/Weiliang Zhang Group/Data/Segment/地方规范性文件'+str(j+1)+'分词结果.xlsx')
        textContext.append(data)
        endSpecial=time.time()
        print('已获得地方规范性文件'+str(j+1)+'的数据，用时'+str(round(endSpecial-startSpecial,2))+'秒')
    data=pd.concat(textContext,ignore_index=True)
    data.fillna(value=' ',inplace=True)
    del textContext
    
    #向量化模型
    keyNum=5000
    startSpecial=time.time()
    vectorModel,data_Stand=tool.vectorData(keyNum,data)
    endSpecial=time.time()
    print('用时'+str(round(endSpecial-startSpecial,2))+'秒')
    del data
    
    #应用LDA模型获取主题
    plexs=[]
    scores=[]
    for j in range(30,34):
        startSpecial=time.time()
        LDAModel=np.load('../data/model/LDAModel_local'+str(j)+'.npy',allow_pickle=True)
        LDAModel=LDAModel[0]
        topic=tool.getTopicsWithModel(25,vectorModel,LDAModel)
        pd.DataFrame(topic).to_excel('../data/主题词/地方规范性主题'+str(j)+'.xlsx',index=False,columns=None)
        plexs.append(LDAModel.perplexity(data_Stand))
        scores.append(LDAModel.score(data_Stand))
        endSpecial=time.time()
        print('处理主题数为'+str(j+1)+'的数据，用时'+str(round(endSpecial-startSpecial,2))+'秒')
        del LDAModel

