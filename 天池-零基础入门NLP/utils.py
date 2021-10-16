import fasttext
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import pandas as pd
import copy
import tempfile
import shutil
# 将各个参数的取值进行排列组合，例如tuned_parameters的示例中，会产生2*4*4*2=64种组合
def get_gridsearch_params(param_grid):
    params_combination = [dict()]  # 用于存放所有可能的参数组合
    for k, v_list in param_grid.items():
        tmp = [{k: v} for v in v_list]
        n = len(params_combination)
        # params_combination = params_combination*len(tmp)  # 浅拷贝，有问题
        copy_params = [copy.deepcopy(params_combination) for _ in range(len(tmp))] 
        params_combination = sum(copy_params, [])
        _ = [params_combination[i*n+k].update(tmp[i]) for k in range(n) for i in range(len(tmp))]
    return params_combination

# 使用k折交叉验证，得到最后的score，保存最佳score以及其对应的那组参数
# 输入分别是训练数据帧，要搜索的参数，用于交叉验证的KFold对象，最佳score评价指标，几分类
def get_KFold_scores(df, params, kf, metric, n_classes):
    metric_score = 0.0

    for train_idx, val_idx in kf.split(df):
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]

        tmpdir = tempfile.mkdtemp() #因为fasttext的训练时读入一个训练数据所在的目录或文件，所以这里用交叉验证集开一个临时目录/文件
        tmp_train_file = tmpdir + '/train.txt'
        df_train.to_csv(tmp_train_file, sep='\t', index=False, header=None, encoding='UTF-8')  # 不要表头

        fast_model = fasttext.train_supervised(tmp_train_file, label_prefix='__label__', thread=3, **params) #训练，传入参数
        
        #用训练好的模型做评估预测
        predicted = fast_model.predict(df_val[0].tolist())  # ([label...], [probs...])
        y_val_pred = [int(label[0][-1:]) for label in predicted[0]]  # label[0]  __label__0
        y_val = [int(cls[-1:]) for cls in df_val[1]]

        score = get_metrics(y_val, y_val_pred, n_classes)[metric]
        metric_score += score #累计在不同的训练集上的score，用于计算在整个交叉验证集上平均分
        shutil.rmtree(tmpdir, ignore_errors=True) #删除临时训练数据文件

    print('平均分:', metric_score / kf.n_splits)
    return metric_score / kf.n_splits

# 网格搜索+交叉验证
# 输入分别是训练数据帧，要搜索的参数，最佳score评价指标，交叉验证要做几折
def my_gridsearch_cv(df, param_grid, metrics, kfold=10):
    n_classes = len(np.unique(df[1]))
    print('n_classes', n_classes)

    kf = KFold(n_splits=kfold)  # k折交叉验证

    params_combination = get_gridsearch_params(param_grid) # 获取参数的各种排列组合

    best_score = 0.0
    best_params = dict()
    for params in params_combination:
        avg_score = get_KFold_scores(df, params, kf, metrics, n_classes)
        if avg_score > best_score:
            best_score = avg_score
            best_params = copy.deepcopy(params)

    return best_score, best_params