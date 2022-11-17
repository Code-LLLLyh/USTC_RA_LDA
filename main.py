# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:26:16 2022

@author: 92853
"""
from utils.apply_LDA import applyLDA 
from utils.tools_LDA import tools_LDA as toolLDA
import pandas as pd
import time 
import numpy as np

if __name__ == '__main__':
    '''
    LDA模型应用的汇总,使用方法In[1]+In[2]+In[i],i为想要进行的操作
    '''
 # In[1] 
  
    #先加载两类函数
    tool=toolLDA()
    apply=applyLDA()
    
 # In[2]  
    '''
    获取数据
    '''
    textContext=[]
    for j in range(0,10):
        startSpecial=time.time()
        #路径注意修改
        data=pd.read_excel('data/province/地方规范性文件_省级'+str(j+1)+'.xlsx')
        textContext.append(data)
        endSpecial=time.time()
        print('已获得地方规范性文件'+str(j+1)+'的数据，用时'+str(round(endSpecial-startSpecial,2))+'秒')
    data=pd.concat(textContext,ignore_index=True)
    data.fillna(value=' ',inplace=True)
    del textContext
    
    modelPath='data/model/'

# In[3] 
    '''
    第一步，使用数据获得LDA模型并储存
    '''
    #进行LDA模型训练，并保存
    tool.resultLDA(data,5000,25,modelPath,isTopic=True)
    
# In[4]
    '''
    第二步，使用已有的LDA模型，获取模型所分好的主题的内容
    '''
    #向量化模型和数据
    keyNum=5000
    startSpecial=time.time()
    vectorModel,data_Stand=tool.vectorData(keyNum,data)
    endSpecial=time.time()
    print('用时'+str(round(endSpecial-startSpecial,2))+'秒')
    del data
    
    #应用LDA模型获取主题
    topicPath='data/主题词/'
    plexs=[]
    scores=[]
    for j in range(30,34):
        startSpecial=time.time()
        LDAModel=np.load(modelPath+'LDAModel_local'+str(j)+'.npy',allow_pickle=True)
        LDAModel=LDAModel[0]
        topic=tool.getTopicsWithModel(25,vectorModel,LDAModel)
        pd.DataFrame(topic).to_excel(topicPath+'地方规范性主题'+str(j)+'.xlsx',index=False,columns=None)
       # plexs.append(LDAModel.perplexity(data_Stand))
       # scores.append(LDAModel.score(data_Stand))
        endSpecial=time.time()
        print('处理主题数为'+str(j+1)+'的数据，用时'+str(round(endSpecial-startSpecial,2))+'秒')
        del LDAModel

# In[5]
    '''        
    第三步,使用已有的LDA模型筛选数据
    '''
    savePath='data/分类/'
    #获取已经储存好的LDA模型
    lda=np.load(modelPath+'LDAModel_local'+str(34)+'.npy',allow_pickle=True)
    lda=lda[0]
    #向量化模型和数据
    keyNum=5000
    startSpecial=time.time()
    vectorModel,data_Stand=apply.vectorData(keyNum,data)
    endSpecial=time.time()
    print('用时'+str(round(endSpecial-startSpecial,2))+'秒')
    #使每条政策都获得对应的主题
    print('正在获取每个政策主题概率并确定主题')
    startSpecial=time.time()
    topics=lda.transform(data_Stand)
    print('已获得主题概率')
    #给每个政策分配主题
    lda_result=apply.ldaResult(topics,data)
    endSpecial=time.time()
    print('用时'+str(round(endSpecial-startSpecial,2))+'秒')
    del data,data_Stand
    
    #筛选主题
    #筛选经济政策
    print('正在筛选经济政策')
    eco=[1,8,10,14,19,21,24,28,31]
    ecoPolicy=apply.selectLDA(eco,lda_result) 
    ecoPolicy.to_csv(savePath+'经济政策.csv',index=False)
    
# In[5]
    '''
    第三步的扩展：一个特殊的要求，不单单筛选经济政策，而是每一类政策，如34主题数，就是34类政策,34个文件
    '''
    keyNum=5000
    savePath='data/分类/'
    vectorModel,data_Stand=apply.vectorData(keyNum,data)    
    Num=34
    apply.mainAction(Num,data_Stand,data,modelPath,savePath)
    
# In[6]
    '''        
    第四步，使用已有的LDA模型进行数据可视化
    '''
    keyNum=5000
    savePath='data/分类/'
    vectorModel,data_Stand=apply.vectorData(keyNum,data) 
    del data
    savePicPath='data/Pic/'
    print('正在完成数据可视化')
    startSpecial=time.time()
    apply.topicsPic(34,vectorModel,data_Stand,modelPath,savePicPath)
    endSpecial=time.time()
    print('已完成，用时'+str(round(endSpecial-startSpecial,2))+'秒')
    