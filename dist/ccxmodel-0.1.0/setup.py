from distutils.core import setup

setup(
    name='ccxmodel',
    version='0.1.0',
    packages=['ccxmodel'],
    url='2017-08-16',
    license='ccx',
    author='liyingkun',
    author_email='liyingkun@ccx.cn',
    description='中诚信征信机器学习平台',
    package_data={'': ['conf/ccxboost.conf', 'conf/ccxgbm.conf', 'conf/ccxrf.conf']},
    # data_files=[('ccxmodel/conf', ['ccxmodel/conf/ccxboost.conf', 'ccxmodel/conf/ccxgbm.conf', 'ccxmodel/conf/ccxrf.conf'])]
)
