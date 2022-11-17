
****   
# 代码说明
本项目主要进行的任务为对地方规范性文件（政策）进行LDA聚类，之后进行分类

# 运行环境
- python 3.7，其余见requirements.txt 

***
# 数据说明  
本代码使用的数据保密，均已经过整理，具体格式为
***
# 文件说明
## 代码部分
### main.py 主要运行程序 
重要的变量：各类path，决定了储存位置
- 块[1]和块[2]是必须运行的
- 块[3]主要是生成模型的代码
- 块[4]使用已经获得的模型生成主题文件，如主题数为30，就生成一个xlsx文件，30行，每行为一个主题的25个重要关键词
- 块[5]对每一条政策进行主题分类，之后筛选出政策来 eco 为重要变量
- 块[5]扩，上一步的扩展，筛选每一个主题的政策，生成对应文件，如主题数为30，就生成30个文件
- 块[6],LDA模型和数据可视化，主要生成一个二维平面图，展示了各个主题之间的距离

### utils.apply_LDA.py
主要进行了块[5],块[5]扩和块[6]的工作

### utils.apply_LDA.py
主要进行了块[3]和块[4]的工作

## 文件部分
### model
储存模型的文件夹
### Pic
储存可视化文件的文件夹
### province
储存省级地方规范性文件数据的文件夹
### 分类
储存筛选出来的数据的文件夹
### 地方
储存全部地方规范性文件数据的文件夹


