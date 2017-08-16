# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:07:12 2017

@author: liyin
"""

'''
中诚信征信机器学习平台：让风控更简单 让模型更精确
机器学习平台使用演示
目前共封装了三个算法,均只支持二分类模型：
    xgboost
    lightgbm
    randomforest（sklearn 0.18.2版本下）
    
支持功能：
    支持环境：windows 10/7 64位 linux下centos7
    用户可通过配置文件自定义参数配置
    自动网格搜索+交叉验证+模型训练+模型评价

不支持和待优化功能：
    数据的预处理
    模型报告的自动生成
    
'''

'''
使用前环境准备：
    1.安装anaconda-python3.6版本
    2.安装xgboost
    3.安装lightgbm
    4.安装configparser
    5.安装ccxmodel包
安装方法：
    1.联网情况下，pip install module_name
    2.离线情况下，pip intall xxxx.whl
    3.ccxmodel安装方法，目录切换到ccxmodel安装目录下（setup.py所在目录）python setup.py install
'''

# 环境安装配置完成后，即可使用 
# 使用演示
from ccxmodel.modelmain import ModelMain

# 0. 切换项目的工作目录为本机已存在的文件夹（很重要）
dir_path=r'C:\\Users\\liyin\\Desktop\\20170620_tn\\0620_base\\model\\' #你自己的工作目录

from ccxmodel.modelconf import CcxBoostConf,CcxGbmConf,CcxRFConf
cbc=CcxBoostConf(ModelMain.CONF_PATH_Boost)
cbc.set_projectdir(dir_path)

cgc=CcxGbmConf(ModelMain.CONF_PATH_Gbm)
cgc.set_projectdir(dir_path)

crc=CcxRFConf(ModelMain.CONF_PATH_RF)
crc.set_projectdir(dir_path)

crc.get_projectdir() #验证是否修改成功

# 1. 初始化模型
train_path = r'C:\Users\liyin\Desktop\20170620_tn\0620_base\train_base14.csv' #训练集数据路径，可见demo_data文件夹
test_path = r'C:\Users\liyin\Desktop\20170620_tn\0620_base\test_base14.csv' #测试集数据路径，可见demo_data文件夹
index_name = 'contract_id' #数据集唯一索引，有且仅支持一个索引，不支持多个索引
target_name = 'target' #目标变量
modelmain = ModelMain(train_path, test_path, index_name, target_name)

# 2.1 使用随机森林进行建模（参数为默认参数）
modelmain.ccxrf_main()

# 2.2 使用xgboost进行建模（参数为默认参数）
modelmain.ccxboost_main()

# 2.3 使用lightgbm进行建模（参数为默认参数）
modelmain.ccxgbm_main()


#3 用户自定义模型超参数 两种方式，修改配置文件，通过代码修改默认配置文件

 #方式一：修改配置文件，并将修改后的配置文件，作为参数传递进模型
modelmain.ccxrf_main(r'C:\Users\liyin\Desktop\ccxmodel\ccxmodel\conf\ccxrf.conf')


 #方式二：程序修改模型的超参数
from ccxmodel.modelconf import CcxBoostConf
cbc=CcxBoostConf(ModelMain.CONF_PATH_Boost)
cbc.set_params('max_depth','4,6') #设置模型超参数,每个模型可调节的超参数不一样，见下文
cbc.get_param('XGB_PARAMS') #查看修改后的超参数列表

modelmain.ccxboost_main() #使用修改后的参数进行建模

cbc.reset() #恢复参数为模型初始的默认配置

#############
'''
三个模型可调节的超参数
！参数的默认值设置希望大家帮忙，依据之前的建模经验，给出一些经验上的较优值
！参数的默认值设置希望大家帮忙，依据之前的建模经验，给出一些经验上的较优值
！参数的默认值设置希望大家帮忙，依据之前的建模经验，给出一些经验上的较优值

设置随机森林的模型超参数
        :param key:
            调节参数
            n_estimators：取值范围[1,inf]的整数 默认100
            max_depth：取值范围整数或None None表示不限制树的深度
            max_features：取值范围0-1的小数 默认0.5
            min_samples_split：取值范围0-1的小数 默认0.1
            min_samples_leaf：取值范围0-1的小数 默认0.01
            bootstrap：取值范围为 True,False
            criterion：取值范围 gini,entropy
        :param value: 用户自定义的值，值的取值范围在上述key的取值范围内
        
设置xgboost模型的超参数
        :param key:
            调节参数
            eta :学习率 默认0.3
            max_depth ：最大树深度 建议取值范围3-10的整型数据 默认4
            subsample ：行抽样比例 0.5
            colsample_bytree ：列抽样比例 0.8
            min_child_weight ：取值范围1-INF 建议范围 1-100 默认2
            gamma ：越大模型越保守
            lambda ：L2正则项参数 越大模型越保守
        :param value: 用户自定义的值，值的取值范围在上述key的取值范围内
        
设置lightgbm模型的超参数
        :param key:
            调节参数
            colsample_bytree :列抽样比例
            learning_rate ：学习率
            min_child_weight ：越大模型越保守
            min_split_gain ：样本划分带来的最小收益
            num_leaves ：叶子节点总数 和max_depth的转换关系为 2**max_depth -1
            reg_lambda :L2正则项
            subsample ：行抽样比例
        :param value: 用户自定义的值，值的取值范围在上述key的取值范围内
'''
