from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from ccxmodel.modelinterface import CcxModel
from ccxmodel.modelutil import ModelUtil


class CcxRF(CcxModel):
    def __init__(self, model_name):
        super(CcxRF, self).__init__(model_name)

    # 1.数据加载
    def model_data(self, train, fillvalue):
        # 数据准备 reference 这个参数需要注意，用于使得测试集与训练集数据结构一致
        ddf = train.fillna(value=fillvalue)
        return ddf

    '''交叉验证+网格搜索'''

    # 此处的train为DataFrame，且为原始数据
    def model_cv(self, train, x_col, y_col, param_grid, nfold=5, message='cv_allcol_1'):
        message = 'model_cv_' + message

        # 使用GridSearchCV 进行网格寻优
        estimator = RandomForestClassifier()

        RF = GridSearchCV(estimator, param_grid, cv=nfold, n_jobs=-1, scoring='roc_auc', verbose=10)

        RF.fit(train[x_col], train[y_col])

        # re = RF.cv_results_
        # dd = pd.DataFrame(re)

        # dd.to_csv((message + '.csv'))

        return RF

    # 3.最优参数的选择

    def get_bstpram(self, gbm):
        re = pd.DataFrame(gbm.cv_results_)
        re['gap'] = np.round(re['mean_train_score'] - re['mean_test_score'], 3)
        re_ = re.query('0.005<=gap<=0.02')

        if len(re_) > 0:
            re_ = re_.sort_values('mean_test_score', ascending=False)
            param = re_.iloc[0, :]['params']
            return param
        else:
            ipos = np.argmax((re['mean_train_score'] + re['mean_test_score'])
                             / np.round(re['mean_train_score'] - re['mean_test_score'], 3))
            print('ipos', ipos, np.round(re['mean_train_score'] - re['mean_test_score'], 3))
            param = re.iloc[ipos, :]['params']

            return param

    def model_train(self, train, x_col, y_col, test, params):
        '''
        模型训练
        '''

        estimator = RandomForestClassifier(
            max_depth=params['max_depth'],
            max_features=params['max_features'],
            n_estimators=params['n_estimators'],
            criterion=params['criterion'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            bootstrap=params['bootstrap'],
            n_jobs=-1,
            random_state=701,
        )
        bst = estimator.fit(train[x_col], train[y_col])
        train_auc = ModelUtil.AUC(bst.predict_proba(train[x_col])[:, 1], train[y_col])
        test_auc = ModelUtil.AUC(bst.predict_proba(test[x_col])[:, 1], test[y_col])
        print('train:%s \t test:%s' % (train_auc, test_auc))

        return bst

    def get_importance_var(self, bst, train):
        '''
        获取进入模型的重要变量
        '''

        re = pd.DataFrame({'Feature_Name': train.columns,
                           'gain': bst.feature_importances_})

        re = re.sort_values('gain', ascending=False)
        re = re.query('gain >0')

        re = re.assign(
            pct_importance=lambda x: x['gain'].apply(lambda s: str(np.round(s / np.sum(x['gain']) * 100, 2)) + '%'))
        print('重要变量的个数：%d' % len(re))
        return re

    def model_predict(self, bst, train, test, x_col, y_col, MU, message='data_id'):
        train_pred_y_xg = bst.predict_proba(train[x_col])[:, 1]
        test_pred_y_xg = bst.predict_proba(test[x_col])[:, 1]

        train_report = classification_report(train[y_col], train_pred_y_xg > 0.5)
        test_report = classification_report(test[y_col], test_pred_y_xg > 0.5)
        print('训练集模型报告：\n', train_report)
        print('测试集模型报告：\n', test_report)

        # 初始化日志文件，保存模型结果
        message_ = 'model_report_' + str(message)
        print(message_)
        infoLogger = MU.model_infologger(message_)
        infoLogger.info('train_report:\n%s' % train_report)
        infoLogger.info('test_report:\n%s' % test_report)

        ks_train = ModelUtil.ks(train_pred_y_xg, train[y_col])

        ks_test = ModelUtil.ks(test_pred_y_xg, test[y_col])

        print('ks_train: %f,ks_test：%f' % (ks_train, ks_test))
        infoLogger.info('ks_train: %f,ks_test：%f \n\n' % (ks_train, ks_test))

        return train_pred_y_xg, test_pred_y_xg

    def get_modelpredict_re(self, test_index, test_pred):
        re = pd.DataFrame([test_index, test_pred]).T
        re.rename(columns={'Unnamed 0': 'P_value'}, inplace=True)
        return re
