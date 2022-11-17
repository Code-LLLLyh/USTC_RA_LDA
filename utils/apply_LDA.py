# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:20:32 2022

@author: 92853
"""

from sklearn.feature_extraction.text import  CountVectorizer
import time 
import pandas as pd
import numpy as np
import pyLDAvis
import pyLDAvis.sklearn

class applyLDA():
    '''
    将已经得到的模型进行应用，获得对应数据的主题、困惑度等
    '''
    def __init__(self):
        self.workContext='应用工具包'
    
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
    
    def ldaResult(self,topics,data):
        '''
        使用lda模型给政策分类
        :param topics array 每一个政策对应各个主题的概率
        :param data dataframe 数据
        :return data_train dataframe 增加了“概率最大的主题序号”和“各个主题对应概率”的数据
        '''
        data_train=data.copy()
        topic = []
        for t in topics:
            topic.append("Topic #"+str(list(t).index(np.max(t))))
        data_train['概率最大的主题序号']=topic
        data_train['每个主题对应概率']=list(topics)
        
        return data_train
    
    def selectLDA(self,idNum,lda_result):
        '''
        筛选LDA中需要的主题的数据
        :param list 主题序号的list，元素为num
        :param lda_result dataframe 在原数据上适配了对应主题序号和概率的数据
        :return data dataframe 筛选出来的数据
        '''
        startSpecial=time.time()

        #先将序号转换成文字列表
        ecoList=[]
        for a in idNum:
            name="Topic #"+str(a)
            ecoList.append(name)
        #筛选数据
        selectLDA=[]
        for eco in ecoList:
            resultLDA=lda_result.loc[(lda_result['概率最大的主题序号']==eco)]
            selectLDA.append(resultLDA)
        #合并
        data=pd.concat(selectLDA,ignore_index=True)
        data.reset_index(drop=True,inplace=True)
        
        endSpecial=time.time()
        print('用时'+str(round(endSpecial-startSpecial,2))+'秒')
        
        return data
    
    def mainAction(self,Num,data_Stand,data,modelPath,savePath):
        '''
        使用已有的模型给每条政策划分主题，即给全部政策分主题后保存
        :param Num int 主题数
        :param data_Stand CountVectorizer 词向量化的数据
        :param data dataframe 原始数据
        :param modelPath str 储存模型的路径
        :param savePath str 主题分类保存的地址
        '''
        #获取已经储存好的LDA模型
        lda=np.load(modelPath+'LDAModel_local'+str(Num)+'.npy',allow_pickle=True)
        lda=lda[0]
        
        #获得主题
        print('正在获取每个政策主题概率并确定主题')
        startSpecial=time.time()
        topics=lda.transform(data_Stand)
        print('已获得主题概率')
        #给每个政策分配主题
        lda_result=tool.ldaResult(topics,data)
        endSpecial=time.time()
        print('分配主题结束，用时'+str(round(endSpecial-startSpecial,2))+'秒')
        
        #筛选主题
        #筛选经济政策
        print('正在筛选政策')
        for i in range(0,34):
            eco=[i]    
            policy=tool.selectLDA(eco,lda_result)
            #储存数据
            policy.to_csv(savePath+'LDA主题数_'+str(Num)+'_Topic_#'+str(i)+'.csv',index=False)

    def topicsPic(self,Num,vectorModel,data_Stand,modelPath,savePath):
        '''
        可视化分类主题
        :param Num int 主题数
        :param vectorModel CountVectorizer 词向量模型
        :param data_Stand CountVectorizer 词向量化的数据
        :param modelPath str 储存模型的路径
        :param savePath str 可视化结果保存的地址
        '''
        #获取已经储存好的LDA模型
        lda=np.load(modelPath+'LDAModel_local'+str(Num)+'.npy',allow_pickle=True)
        lda=lda[0]
        #绘制图像
        pic = pyLDAvis.sklearn.prepare(lda,data_Stand,vectorModel,n_jobs=1)
        #pyLDAvis.display(pic)
        pyLDAvis.save_html(pic,savePath+'lda_Topics'+str(Num)+'.html')
        #pyLDAvis.display(pic)

    
if __name__ == '__main__':

    tool=applyLDA()
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
    print('正在词向量化数据')
    keyNum=5000
    startSpecial=time.time()
    vectorModel,data_Stand=tool.vectorData(keyNum,data)
    endSpecial=time.time()
    print('用时'+str(round(endSpecial-startSpecial,2))+'秒')

    ##数据可视化
    del data
    print('正在完成数据可视化')
    startSpecial=time.time()
    tool.topicsPic(33,vectorModel,data_Stand)
    endSpecial=time.time()
    print('已完成，用时'+str(round(endSpecial-startSpecial,2))+'秒')
    
    ##这部分是用来跑全部政策分类的代码
    #tool.mainAction(33,data_Stand,data)



    ##这部分是用来读主题数为33的经济和环保政策分类的代码
    '''
    #获取已经储存好的LDA模型
    lda=np.load('../data/model/LDAModel_local'+str(33)+'.npy',allow_pickle=True)
    lda=lda[0]
    #获得主题
    print('正在获取每个政策主题概率并确定主题')
    startSpecial=time.time()
    topics=lda.transform(data_Stand)
    print('已获得主题概率')
    #给每个政策分配主题
    lda_result=tool.ldaResult(topics,data)
    endSpecial=time.time()
    print('用时'+str(round(endSpecial-startSpecial,2))+'秒')
    del data,data_Stand
    
    #筛选主题
    #筛选经济政策
    print('正在筛选经济政策')
    eco=[1,8,10,14,19,21,24,28,31]
    ecoPolicy=tool.selectLDA(eco,lda_result)
    #筛选环保政策
    print('正在筛选环保政策')
    env=[12]
    envPolicy=tool.selectLDA(env,lda_result)
    
    #储存数据
    #ecoPolicy.to_excel('../data/分类/经济政策.xlsx',index=False)
    #envPolicy.to_excel('../data/分类/环保政策.xlsx',index=False)
    ecoPolicy.to_csv('../data/分类/经济政策.csv',index=False)
    envPolicy.to_csv('../data/分类/环保政策.csv',index=False)
    
    #抽样查看分类，并保存
    ecoPolicy=pd.read_csv('../data/分类/经济政策.csv')
    data_train=ecoPolicy.sample(frac=0.1).copy()
    data_train.to_excel('../data/分类/经济抽样.xlsx',index=False)
    '''
