import os
import pandas as pd
import time
import logging
import datetime
import numpy as np
import pickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from ccxmodel.modelconf import ModelConf

plt.switch_backend('agg')  # 解决matplotlib在Linux下图片不能显示的报错问题

class ModelUtil(object):
    '''
    模型工具类函数，主要用于封装三个模型中的共性特征
    '''

    def __init__(self, conf_path):
        self.root_path = ModelConf(conf_path).get_projectdir()
        self.root_path = self.model_result_path()
        print(self.root_path)

    def model_result_path(self):
        timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
        filename = 'model' + timestamp
        path = self.root_path + filename
        if os.path.exists(path):
            return path
        else:
            os.mkdir(path)
            return path

    def model_infologger(self, _message):
        path = self.root_path + '/modellog'
        if os.path.exists(path):
            os.chdir(path)
        else:
            os.mkdir(path)
            os.chdir(path)

        format = '%(asctime)s - %(filename)s - [line:%(lineno)d] - %(levelname)s:\n %(message)s'
        curDate = datetime.date.today() - datetime.timedelta(days=0)
        infoLogName = r'%s_info_%s.log' % (_message, curDate)

        formatter = logging.Formatter(format)

        infoLogger = logging.getLogger('%s_info_%s.log' % (_message, curDate))

        #  这里进行判断，如果logger.handlers列表为空，则添加，否则，直接去写日志
        if not infoLogger.handlers:
            infoLogger.setLevel(logging.INFO)

            infoHandler = logging.FileHandler(infoLogName, 'a')
            infoHandler.setLevel(logging.INFO)
            infoHandler.setFormatter(formatter)
            infoLogger.addHandler(infoHandler)

        os.chdir(self.root_path)

        return infoLogger

    def save_bstmodel(self, bst, mess):
        path = self.root_path + '/modeltxt'
        if os.path.exists(path):
            os.chdir(path)
        else:
            os.mkdir(path)
            os.chdir(path)

        curDate = datetime.date.today() - datetime.timedelta(days=0)
        path_1 = 'model_' + mess + '_' + str(curDate) + '.txt'
        with open(path_1, 'wb') as f:
            pickle.dump(bst, f)
        os.chdir(self.root_path)
        print('模型保存成功 文件路径名：%s' % (path + '/' + path_1))
        return path + '/' + path_1

    @staticmethod
    def load_bstmodel(path):
        with open(path, 'rb') as f:
            bst = pickle.load(f)
        return bst

    @staticmethod
    def ks(y_pred, y_true):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        ks = np.max(tpr - fpr)
        return ks

    @staticmethod
    def AUC(y_pred, y_true):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        return roc_auc

    def plot_ks_line(self, y_true, y_pred, title='ks-line', detail=False):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

        plt.plot(tpr, label='tpr-line')
        plt.plot(fpr, label='fpr-line')
        plt.plot(tpr - fpr, label='KS-line')
        # 设置x的坐标轴为0-1范围
        plt.xticks(np.arange(0, len(tpr), len(tpr) // 10), np.arange(0, 1.1, 0.1))

        # 添加标注
        x0 = np.argmax(tpr - fpr)
        y0 = np.max(tpr - fpr)
        plt.scatter(x0, y0, color='black')  # 显示一个点
        z0 = thresholds[x0]  # ks值对应的阈值
        plt.text(x0 - 2, y0 - 0.12, ('(ks: %.4f,\n th: %.4f)' % (y0, z0)))

        if detail:
            # plt.plot([x0,x0],[0,y0],'b--',label=('thresholds=%.4f'%z0)) #在点到x轴画出垂直线
            # plt.plot([0,x0],[y0,y0],'r--',label=('ks=%.4f'%y0)) #在点到y轴画出垂直线
            plt.plot(thresholds[1:], label='thresholds')
            t0 = thresholds[np.argmin(np.abs(thresholds - 0.5))]
            t1 = list(thresholds).index(t0)
            plt.scatter(t1, t0, color='black')
            plt.plot([t1, t1], [0, t0])
            plt.text(t1 + 2, t0, 'thresholds≈0.5')

            tpr0 = tpr[t1]
            plt.scatter(t1, tpr0, color='black')
            plt.text(t1 + 2, tpr0, ('tpr=%.4f' % tpr0))

            fpr0 = fpr[t1]
            plt.scatter(t1, fpr0, color='black')
            plt.text(t1 + 2, fpr0, ('fpr=%.4f' % fpr0))

        plt.legend(loc='upper left')
        plt.title(title)
        fig_path = self.save_figure(plt, title)
        # plt.show()
        plt.close()
        return fig_path

    '''
    封装一个函数：绘制ROC曲线
    '''

    def plot_roc_line(self, y_true, y_pred, title='ROC-line'):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        ks = np.max(tpr - fpr)
        plt.plot(fpr, tpr)  # ,label=('auc= %.4f'%roc_auc)
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
        plt.text(0.7, 0.45, ('auc= %.4f \nks  = %.4f' % (roc_auc, ks)))

        plt.title(title)
        fig_path = self.save_figure(plt, title)
        # plt.show()
        plt.close()
        return fig_path

    @staticmethod
    def load_data(path, *args):
        data = pd.read_csv(path, *args)
        return data

    def save_data(self, data, data_name, index=False):
        curDate = datetime.date.today() - datetime.timedelta(days=0)
        path = self.root_path + '/modeldata'
        if os.path.exists(path):
            os.chdir(path)
        else:
            os.mkdir(path)
            os.chdir(path)

        data_name = 'd_' + str(curDate) + '_' + data_name
        data.to_csv(data_name, index=index)

        os.chdir(self.root_path)
        print('数据保存成功:%s' % (path + '/' + data_name))
        return path + '/' + data_name

    def save_figure(self, fig, fig_name):
        curDate = datetime.date.today() - datetime.timedelta(days=0)
        path = self.root_path + '/modelfig'
        if os.path.exists(path):
            os.chdir(path)
        else:
            os.mkdir(path)
            os.chdir(path)

        fig_name = 'd_' + str(curDate) + '_' + fig_name
        fig.savefig(fig_name)
        print('图片保存成功:%s' % (path + '/' + fig_name + '.png'))
        os.chdir(self.root_path)
        return path + '/' + fig_name + '.png'

    @staticmethod
    def write_path(file, path_list):
        with open(file, 'w') as f:
            f.writelines([line + '\n' for line in path_list])
            f.write('\n')
        print('结果路径写入到%s文件中' % file)

    ###特例独行的一个函数，用于解决fit产生的空文件夹这个bug
    @staticmethod
    def rmemptydir(rootpath):
        dirs = os.listdir(rootpath)
        for dirpath in dirs:
            x = os.path.join(rootpath, dirpath)
            if os.path.isdir(x) and not os.listdir(x):
                try:
                    os.rmdir(x)
                except:
                    print('文件夹%s删除失败' % x)

    @staticmethod
    def splitdata(data, x_columns, y_columns, testsize=0.3, rdm_state=701):
        X = data[x_columns]
        Y = data[y_columns]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testsize,
                                                            random_state=rdm_state, stratify=Y)

        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)
        return train, test


if __name__ == '__main__':
    path = r'C:\Users\liyin\Desktop\ccxrf\ccxrf\ccxrf.conf'
    MU = ModelUtil(path)

    xx = ModelUtil.load_data(r'C:\Users\liyin\Desktop\20170620_tn\0620_base\train_base14.csv')
    MU.save_data(xx, 'lyk0815.csv')
