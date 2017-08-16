import xgboost as xgb
from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve

from ccxmodel.modelinterface import CcxModel
from ccxmodel.modelutil import ModelUtil


class CcxBoost(CcxModel):
    def __init__(self, model_name):
        super(CcxBoost, self).__init__(model_name)

    def model_data(self, df, x_colnames, y_colnames, miss=np.nan):
        # 数据准备
        ddf = xgb.DMatrix(df[x_colnames], label=df[y_colnames], missing=miss)
        return ddf

    def model_cv(self, dtrain, param_grid, num_boost_round, mu, nfold=5,  message='cv_allcol_1'):
        # 初始化日志文件
        message = 'model_cv_' + message
        infoLogger = mu.model_infologger(message)

        ''' 定制化评价函数'''

        def kserro(preds, dtrain):
            fpr, tpr, thresholds = roc_curve(dtrain.get_label(), preds)
            ks = np.max(tpr - fpr)
            return 'ks', ks

        params = list(ParameterGrid(param_grid))
        i = 0
        result = []
        for param in params:
            re = xgb.cv(param, dtrain, num_boost_round, nfold=nfold, verbose_eval=True)
            param['max_test_auc'] = re.iloc[np.argmax(re['test-auc-mean'])]['test-auc-mean']
            param['max_train_auc'] = re.iloc[np.argmax(re['test-auc-mean'])]['train-auc-mean']
            # 0624 ks的加入，特别说明，auc和ks各自取最大 feval=kserro,
            # param['max_test_ks'] = re.iloc[np.argmax(re['test-ks-mean'])]['test-ks-mean']
            # param['max_train_ks'] = re.iloc[np.argmax(re['test-ks-mean'])]['train-ks-mean']

            param['num_best_round'] = re['test-auc-mean'].diff().abs().argmin() + 1  # 一阶差分不会增加，迭代次数不能带来实质的提高
            param['num_maxtest_round'] = np.argmax(re['test-auc-mean']) + 1

            param['num_boost_round'] = num_boost_round
            result.append(param)

            infoLogger.info(param)
            infoLogger.info('max_test_auc:%s' % param['max_test_auc'])
            infoLogger.info('num_maxtest_round:%d' % param['num_maxtest_round'])
            infoLogger.info('max_train_auc:%s' % param['max_train_auc'])
            infoLogger.info('num_best_round:%d' % param['num_best_round'])

            i += 1
            infoLogger.info('<<<总计选参：%d,已运行到：第 %d 个参数>>>\n\n' % (len(params), i))

            print('\n\n<<<总计选参：%d,已运行到：%d 个参数>>>\n\n' % (len(params), i))

            # save_data(pd.DataFrame(result), message + '.csv')
            # pd.DataFrame(result).to_csv((message + '.csv'))

        result = pd.DataFrame(result)

        return result

    def get_bstpram(self, re, method='defualt'):
        if np.argmax(re['max_test_auc']) == np.argmin(re['max_train_auc'] - re['max_test_auc']):
            print('warning:two method have eqaul bst result.')

        re['gap'] = np.round(re['max_train_auc'] - re['max_test_auc'], 3)
        re_ = re.query('0.005<=gap<=0.02')
        if len(re_) > 0:
            re_ = re_.sort_values('max_test_auc', ascending=False)
            param = dict(re_.iloc[0, :])
            num_round = param['num_maxtest_round']
            return param, num_round
        else:
            ipos = np.argmax((re['max_train_auc'] + re['max_test_auc'])
                             / np.round(re['max_train_auc'] - re['max_test_auc'], 3))
            param = dict(re.iloc[ipos, :])
            num_round = param['num_maxtest_round']
            return param, num_round

    def model_train(self, dtrain_text, dtest_text, param, num_round):
        '''
        模型训练
        '''

        def kserro(preds, dtrain):
            fpr, tpr, thresholds = roc_curve(dtrain.get_label(), preds)
            ks = np.max(tpr - fpr)
            return 'ks', ks

        watchlist = [(dtest_text, 'test'), (dtrain_text, 'train')]
        bst = xgb.train(param, dtrain_text, num_round, watchlist, verbose_eval=True)
        # feval=kserro,
        return bst

    def get_importance_var(self, bst):
        '''
        获取进入模型的重要变量
        '''
        re = pd.Series(bst.get_score(importance_type='gain')).sort_values(ascending=False)
        re = pd.DataFrame(re, columns=['gain']).reset_index()
        re.columns = ['Feature_Name', 'gain']
        re = re.assign(
            pct_importance=lambda x: x['gain'].apply(lambda s: str(np.round(s / np.sum(x['gain']) * 100, 2)) + '%'))
        print('重要变量的个数：%d' % len(re))
        return re

    def model_predict(self, bst, dtrain_text, dtest_text, MU, message='data_id'):
        train_pred_y_xg = bst.predict(dtrain_text)
        test_pred_y_xg = bst.predict(dtest_text)

        train_report = classification_report(dtrain_text.get_label(), train_pred_y_xg > 0.5)
        test_report = classification_report(dtest_text.get_label(), test_pred_y_xg > 0.5)
        print('训练集模型报告：\n', train_report)
        print('测试集模型报告：\n', test_report)

        # 初始化日志文件，保存模型结果
        message = 'model_report_' + str(message)
        infoLogger = MU.model_infologger(message)
        infoLogger.info('train_report:\n%s' % train_report)
        infoLogger.info('test_report:\n%s' % test_report)

        ks_train = ModelUtil.ks(train_pred_y_xg, dtrain_text.get_label())

        ks_test = ModelUtil.ks(test_pred_y_xg, dtest_text.get_label())

        print('ks_train: %f,ks_test：%f' % (ks_train, ks_test))
        infoLogger.info('ks_train: %f,ks_test：%f \n\n' % (ks_train, ks_test))

        return train_pred_y_xg, test_pred_y_xg

    def get_modelpredict_re(self, test_index, test_pred):
        re = pd.DataFrame([test_index, test_pred]).T
        re.rename(columns={'Unnamed 0': 'P_value'}, inplace=True)
        return re
