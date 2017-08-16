import configparser


class ModelConf(object):
    '''
    模型配置文件类 用于提供默认的初始化模型参数和自定义的参数设置
    '''

    def __init__(self, conf_path):
        self.conf_path = conf_path
        cf = configparser.ConfigParser()
        cf.read(conf_path)
        self.conf = cf
        self.flag = True

    def __del__(self):
        '''
        当用户对配置文件修改之后，用户关闭页面时，将配置文件恢复为默认的状态
        这个函数待测试
        :return:
        '''
        if self.flag:
            self.reset()
            print('调用结束')

    def reset(self):
        '''
        用于恢复配置文件的默认设置
        :return:
        '''
        self._save_conf(self.conf)

    def get_projectdir(self):
        conf = configparser.ConfigParser()
        conf.read(self.conf_path)
        projectdir = conf.get("DIRECTORY", "project_pt")
        return projectdir

    def _trans(self, x):
        if x.strip() == 'None':
            x = None
        elif x.strip() == 'True':
            x = True
        elif x.strip() == 'False':
            x = False
        else:
            x
        return x

    def get_param(self, conf_section):
        conf = configparser.ConfigParser()
        conf.read(self.conf_path)
        kvs = conf.items(conf_section)

        param = {}
        for (m, n) in kvs:
            n_v = n.split(',')
            new_n_v = []
            for j in n_v:
                try:
                    try:
                        new_n_v.append(int(j))
                    except:
                        new_n_v.append(float(j))
                except:
                    new_n_v.append(self._trans(j))
            param[m] = new_n_v
        return param

    def get_cv(self, sec):
        conf = configparser.ConfigParser()
        conf.read(self.conf_path)
        cv = conf.getint(sec, "cv")
        return cv

    def get_cvmess(self, sec):
        conf = configparser.ConfigParser()
        conf.read(self.conf_path)
        cvmess = conf.get(sec, "cv_mess")
        return cvmess

    def get_num_round(self, sec):
        conf = configparser.ConfigParser()
        conf.read(self.conf_path)
        num_round = conf.getint(sec, "num_round")
        return num_round

    def _save_conf(self, conf):
        with open(self.conf_path, 'w') as f:
            conf.write(f)

    def set_projectdir(self, proj_path):
        conf = configparser.ConfigParser()
        conf.read(self.conf_path)
        conf.set("DIRECTORY", "project_pt", proj_path)
        self._save_conf(conf)
        self.flag = False

    def set_cv(self, sec, cv):
        conf = configparser.ConfigParser()
        conf.read(self.conf_path)
        conf.set(sec, "cv", cv)
        self._save_conf(conf)

    def set_cvmess(self, sec, cvmess):
        conf = configparser.ConfigParser()
        conf.read(self.conf_path)
        conf.set(sec, "cv_mess", cvmess)
        self._save_conf(conf)

    def set_num_round(self, sec, num_round):
        conf = configparser.ConfigParser()
        conf.read(self.conf_path)
        conf.set(sec, "num_round", num_round)
        self._save_conf(conf)

    def set_params(self):
        '''
        不同的模型，对应的参数不同，可用于继承并复写
        :return:
        '''
        pass

    pass


class CcxRFConf(ModelConf):
    def __init__(self, conf_path):
        super(CcxRFConf, self).__init__(conf_path)

    def set_params(self, key, value, sec="RF_PARAMS"):
        '''
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
        :param sec: RF_PARAMS
        :return:
        '''
        conf = configparser.ConfigParser()
        conf.read(self.conf_path)
        conf.set(sec, key, value)
        self._save_conf(conf)


class CcxBoostConf(ModelConf):
    def __init__(self, conf_path):
        super(CcxBoostConf, self).__init__(conf_path)

    def set_params(self, key, value, sec="XGB_PARAMS"):
        '''
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
        :param sec: RF_PARAMS
        :return:
        '''
        conf = configparser.ConfigParser()
        conf.read(self.conf_path)
        conf.set(sec, key, value)
        self._save_conf(conf)


class CcxGbmConf(ModelConf):
    def __init__(self, conf_path):
        super(CcxGbmConf, self).__init__(conf_path)

    def set_params(self, key, value, sec="GBM_PARAMS"):
        '''
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
        :param sec: RF_PARAMS
        :return:
        '''
        conf = configparser.ConfigParser()
        conf.read(self.conf_path)
        conf.set(sec, key, value)
        self._save_conf(conf)

