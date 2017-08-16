'''
模型调用所用的主函数
'''
from ccxmodel.ccxboost import CcxBoost
from ccxmodel.ccxgbm import CcxGbm
from ccxmodel.ccxrf import CcxRF
from ccxmodel.modelconf import ModelConf, CcxBoostConf
from ccxmodel.modelutil import ModelUtil
import pandas as pd
import os


class ModelMain(object):
    CONF_PATH = os.path.abspath(os.path.dirname(__file__))
    CONF_PATH = CONF_PATH + '/conf'
    CONF_PATH_RF = CONF_PATH + '/ccxrf.conf'
    CONF_PATH_Boost = CONF_PATH + '/ccxboost.conf'
    CONF_PATH_Gbm = CONF_PATH + '/ccxgbm.conf'

    def __init__(self, train_path, test_path, index_name, target_name):
        self.train_path = train_path
        self.test_path = test_path
        self.index_name = index_name
        self.target_name = target_name

    def ccxrf_main(self, conf_path=CONF_PATH_RF):
        print('配置文件目录：%s' % conf_path)
        Model = CcxRF("ccxrf")
        Conf = ModelConf(conf_path)
        MU = ModelUtil(conf_path)
        conf_sec = "RF_OPTIONS"
        # 1.读取数据
        train = MU.load_data(self.train_path)  # .select_dtypes(exclude=['object'])
        test = MU.load_data(self.test_path)  # .select_dtypes(exclude=['object'])

        object_var = list(train.select_dtypes(include=['object']).columns.values)
        warn_col = [x for x in object_var if x not in [self.index_name]]
        if warn_col:
            print('数据中列名为%s的列,不是数值型数据,请转换为数值型数据或删除后再输入.' % warn_col)

        del_col = [self.index_name, self.target_name] + warn_col
        x_colnames = [x for x in train.columns if x not in del_col]
        y_colnames = self.target_name

        # 2.转换数据格式为模型要求格式
        train = Model.model_data(train, -999)
        test = Model.model_data(test, -999)

        # 解析配置文件，获取网格搜索的调参列表

        param_grid = Conf.get_param("RF_PARAMS")
        print('网格参数集：%s' % param_grid)

        # 用config对象读取配置文件，获取到交叉验证的option参数
        cv = Conf.get_cv(conf_sec)
        cv_mess = Conf.get_cvmess(conf_sec)

        # 网格搜索
        re = Model.model_cv(train, x_colnames, y_colnames, param_grid, nfold=cv, message=cv_mess)
        file_name = cv_mess + '_' + str(cv) + 'FlodCV.csv'
        cv_result_path = MU.save_data(pd.DataFrame(re.cv_results_), file_name, index=True)

        param = Model.get_bstpram(re)

        print('最优参数为%s' % param)
        bst = Model.model_train(train, x_colnames, y_colnames, test, param)
        model_path = MU.save_bstmodel(bst, cv_mess)
        # bst.dump_model('bst_model.txt')

        # 重要变量
        imp_var = Model.get_importance_var(bst, train[x_colnames])
        imp_path = MU.save_data(imp_var, 'importance_var.csv')

        # 模型预测与模型评估
        train_pred_y, test_pred_y = Model.model_predict(bst, train, test, x_colnames, y_colnames, MU, message=cv_mess)
        # 模型预测结果
        pred_path = MU.save_data(Model.get_modelpredict_re(test[self.index_name], test_pred_y),
                                 'test_predict.csv')
        # 画图
        trks_path = MU.plot_ks_line(train[y_colnames], train_pred_y, title=cv_mess + '_train_ks-line')
        trauc_path = MU.plot_roc_line(train[y_colnames], train_pred_y, title=cv_mess + '_train_ROC-line')
        # 注意，现在仅支持测试集有目标变量的，没有的情况需要后期优化时注意
        teks_path = MU.plot_ks_line(test[y_colnames], test_pred_y, title=cv_mess + '_test_ks-line')
        teauc_path = MU.plot_roc_line(test[y_colnames], test_pred_y, title=cv_mess + '_test_ROC-line')

        path_list = [cv_result_path, model_path, imp_path, pred_path, trks_path, trauc_path, teks_path, teauc_path]
        file = r'C:\Users\liyin\Desktop\20170620_tn\0620_base\model\modelpath.txt'
        MU.write_path(file, path_list)
        # print(path_list)

        MU.rmemptydir(Conf.get_projectdir())

    def ccxboost_main(self, conf_path=CONF_PATH_Boost):
        Model = CcxBoost("ccxboost")
        Conf = ModelConf(conf_path)
        MU = ModelUtil(conf_path)
        conf_sec = "XGB_OPTIONS"
        # 1.读取数据
        train = MU.load_data(self.train_path)  # .select_dtypes(exclude=['object'])
        test = MU.load_data(self.test_path)  # .select_dtypes(exclude=['object'])

        object_var = list(train.select_dtypes(include=['object']).columns.values)
        warn_col = [x for x in object_var if x not in [self.index_name]]
        if warn_col:
            print('数据中列名为%s的列,不是数值型数据,请转换为数值型数据或删除后再输入.' % warn_col)

        del_col = [self.index_name, self.target_name] + warn_col
        x_colnames = [x for x in train.columns if x not in del_col]
        y_colnames = self.target_name

        # 2.转换数据格式为模型要求格式
        # 2.转换数据格式为模型要求格式
        dtrain = Model.model_data(train, x_colnames, y_colnames)
        dtest = Model.model_data(test, x_colnames, y_colnames)

        # 解析配置文件，获取网格搜索的调参列表

        param_grid = Conf.get_param("XGB_PARAMS")
        print('网格参数集：%s' % param_grid)

        # 用config对象读取配置文件，获取到交叉验证的option参数
        cv = Conf.get_cv(conf_sec)
        cv_mess = Conf.get_cvmess(conf_sec)
        num_boost_rounds = Conf.get_num_round(conf_sec)

        # 网格搜索
        re = Model.model_cv(dtrain, param_grid, num_boost_rounds, MU, nfold=cv, message=cv_mess)
        file_name = cv_mess + '_' + str(cv) + 'FlodCV.csv'
        cv_result_path = MU.save_data(re, file_name, index=True)

        param, num_round = Model.get_bstpram(re)
        print(param)

        bst = Model.model_train(dtrain, dtest, param, num_round)
        model_path = MU.save_bstmodel(bst, cv_mess)
        # bst.dump_model('bst_model.txt')

        # 重要变量
        imp_var = Model.get_importance_var(bst)
        # plot_imp(bst)
        imp_path = MU.save_data(imp_var, 'importance_var.csv')

        # 模型预测与模型评估
        train_pred_y, test_pred_y = Model.model_predict(bst, dtrain, dtest, MU, message=cv_mess)
        # 模型预测结果
        pred_path = MU.save_data(Model.get_modelpredict_re(test[self.index_name], test_pred_y),
                                 'test_predict.csv')
        # 画图
        trks_path = MU.plot_ks_line(dtrain.get_label(), train_pred_y, title=cv_mess + '_train_ks-line')
        trauc_path = MU.plot_roc_line(dtrain.get_label(), train_pred_y, title=cv_mess + '_train_ROC-line')
        # 注意，现在仅支持测试集有目标变量的，没有的情况需要后期优化时注意
        teks_path = MU.plot_ks_line(dtest.get_label(), test_pred_y, title=cv_mess + '_test_ks-line')
        teauc_path = MU.plot_roc_line(dtest.get_label(), test_pred_y, title=cv_mess + '_test_ROC-line')

        path_list = [cv_result_path, model_path, imp_path, pred_path, trks_path, trauc_path, teks_path, teauc_path]
        file = r'C:\Users\liyin\Desktop\20170620_tn\0620_base\model\modelpath.txt'
        MU.write_path(file, path_list)

    def ccxgbm_main(self, conf_path=CONF_PATH_Gbm):
        Model = CcxGbm("ccxgbm")
        Conf = ModelConf(conf_path)
        MU = ModelUtil(conf_path)
        conf_sec = "GBM_OPTIONS"
        # 1.读取数据
        train = MU.load_data(self.train_path)  # .select_dtypes(exclude=['object'])
        test = MU.load_data(self.test_path)  # .select_dtypes(exclude=['object'])

        object_var = list(train.select_dtypes(include=['object']).columns.values)
        warn_col = [x for x in object_var if x not in [self.index_name]]
        if warn_col:
            print('数据中列名为%s的列,不是数值型数据,请转换为数值型数据或删除后再输入.' % warn_col)

        del_col = [self.index_name, self.target_name] + warn_col
        x_colnames = [x for x in train.columns if x not in del_col]
        y_colnames = self.target_name

        # 2.转换数据格式为模型要求格式
        dtrain = Model.model_data(train, x_colnames, y_colnames)
        dtest = Model.model_data(test, x_colnames, y_colnames)

        # 解析配置文件，获取网格搜索的调参列表
        param_grid = Conf.get_param("GBM_PARAMS")
        print('网格参数集：%s' % param_grid)

        # 用config对象读取配置文件，获取到交叉验证的option参数
        cv = Conf.get_cv(conf_sec)
        cv_mess = Conf.get_cvmess(conf_sec)
        num_boost_rounds = Conf.get_num_round(conf_sec)

        # 网格搜索
        re = Model.model_cv(train, x_colnames, y_colnames, param_grid, num_boost_rounds, nfold=cv, message=cv_mess)
        file_name = cv_mess + '_' + str(cv) + 'FlodCV.csv'
        cv_result_path = MU.save_data(pd.DataFrame(re.cv_results_), file_name, index=True)

        param = Model.get_bstpram(re)
        param = dict(param, **{'boosting_type': 'gbdt',
                               'objective': 'binary',
                               'metric': 'auc'})
        print('最优参数为%s' % param)
        bst = Model.model_train(dtrain, dtest, param, num_boost_rounds)
        model_path = MU.save_bstmodel(bst, cv_mess)
        # bst.dump_model('bst_model.txt')

        # 重要变量
        imp_var = Model.get_importance_var(bst)
        # plot_imp(bst)
        imp_path = MU.save_data(imp_var, 'importance_var.csv')

        # 模型预测与模型评估
        train_pred_y, test_pred_y = Model.model_predict(bst, train, test, x_colnames, y_colnames, MU, message=cv_mess)
        # 模型预测结果
        pred_path = MU.save_data(Model.get_modelpredict_re(test[self.index_name], test_pred_y),
                                 'test_predict.csv')
        # 画图
        trks_path = MU.plot_ks_line(dtrain.get_label(), train_pred_y, title=cv_mess + '_train_ks-line')
        trauc_path = MU.plot_roc_line(dtrain.get_label(), train_pred_y, title=cv_mess + '_train_ROC-line')
        # 注意，现在仅支持测试集有目标变量的，没有的情况需要后期优化时注意
        teks_path = MU.plot_ks_line(dtest.get_label(), test_pred_y, title=cv_mess + '_test_ks-line')
        teauc_path = MU.plot_roc_line(dtest.get_label(), test_pred_y, title=cv_mess + '_test_ROC-line')

        path_list = [cv_result_path, model_path, imp_path, pred_path, trks_path, trauc_path, teks_path, teauc_path]
        file = r'C:\Users\liyin\Desktop\20170620_tn\0620_base\model\modelpath.txt'
        MU.write_path(file, path_list)
        # print(path_list)

        MU.rmemptydir(Conf.get_projectdir())


'''
测试开始
'''

if __name__ == '__main__':
    train_path = r'C:\Users\liyin\Desktop\20170620_tn\0620_base\train_base14.csv'
    test_path = r'C:\Users\liyin\Desktop\20170620_tn\0620_base\test_base14.csv'
    index_name = 'contract_id'
    target_name = 'target'

    # 初始化实例对象
    # modelmain = ModelMain(train_path, test_path, index_name, target_name)
    # modelmain.ccxrf_main(r'C:\Users\liyin\Desktop\ccxrf\ccxrf\ccxrf.conf')
    #
    # modelmain.ccxboost_main(r'C:\Users\liyin\Desktop\ccxboost\ccxboost\ccxboost.conf')
    #
    # modelmain.ccxgbm_main(r'C:\Users\liyin\Desktop\ccxgbm\ccxgbm\ccxgbm.conf')
    #
    # # 修改配置文件参数后，演示用户自定义配置文件参数
    # mc = ModelConf(r'C:\Users\liyin\Desktop\ccxboost\ccxboost\ccxboost.conf')
    # mc.set_cvmess('XGB_OPTIONS', 'lyk')
    # mc.get_cvmess('XGB_OPTIONS')
    # xgbmc = CcxBoostConf(r'C:\Users\liyin\Desktop\ccxboost\ccxboost\ccxboost.conf')
    # xgbmc.set_params('max_depth', '5')
    # xgbmc.get_param('XGB_PARAMS')
    # modelmain.ccxboost_main(r'C:\Users\liyin\Desktop\ccxboost\ccxboost\ccxboost.conf')
    # del mc

    modelmain = ModelMain(train_path, test_path, index_name, target_name)
    modelmain.ccxrf_main()
    modelmain.ccxboost_main()
    modelmain.ccxgbm_main()
