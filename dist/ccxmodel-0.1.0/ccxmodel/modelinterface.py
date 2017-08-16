


class CcxModel(object):
    def __init__(self, model_name):
        self.model_name = model_name

    def model_data(self):
        '''
        加载数据，使其为模型可接收的形式
        :return:
        '''
        pass

    def model_cv(self):
        '''
        模型交叉验证函数，用于选参和选择最优模型
        :return:
        '''
        pass

    def get_bstpram(self):
        '''
        使得模型最优的参数
        :return:
        '''
        pass

    def model_train(self):
        '''
        模型训练
        :return:
        '''
        pass

    def get_importance_var(self):
        '''
        重要变量获取
        :return:
        '''
        pass

    def model_predict(self):
        '''
        模型预测
        :return:
        '''
        pass

    def get_modelpredict_re(self):
        '''
        得到模型预测结果
        :return:
        '''
        pass


