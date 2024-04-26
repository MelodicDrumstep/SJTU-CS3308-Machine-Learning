import os
import sys

import numpy as np
from dataset import get_data,get_HOG,standardize
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

class SVM_info:
    def __init__(self, support_vector_indices, num_support_vectors_each_class, dual_coefs_times_label):
        self.support_vector_indices = support_vector_indices
        self.num_support_vectors_each_class = num_support_vectors_each_class
        self.dual_coefs_times_label = dual_coefs_times_label

# 以下两个函数是用于填充参数列表的辅助函数
def fill_arg_C_list(start, end, step, arg_C_list):
    arg_C_list += np.arange(start, end, step).tolist()

def fill_arg_gamma_list(arg_C_list):
    arg_C_list.append(10)
    arg_C_list.append(100)
    arg_C_list.append(1000)

# 用于训练模型并生成score列表的函数
def training(mode, H_train, Y_train, H_test, Y_test):
    arg_C_list = []
    fill_arg_C_list(0.001, 0.01, 0.001, arg_C_list)
    fill_arg_C_list(0.01, 0.1, 0.02, arg_C_list)
    fill_arg_C_list(0.1, 1, 0.25, arg_C_list)
    fill_arg_C_list(1, 10, 5, arg_C_list)
    fill_arg_gamma_list(arg_C_list)
    # 填充参数列表 arg_C_list

    score_list = []
    SVM_info_list = []
    for argC in arg_C_list:
        # 每轮选定一个参数进行训练
        print("inside SVM")
        # 用于DEBUG的调试信息
        svm = SVC(kernel = mode, C = argC)
        # 生成一个SVC模型，使用的核函数由我给定，正则化参数为argC
        print("C = ", argC)
        svm.fit(H_train, Y_train)
        # 训练过程
        #print(svm.score(H_test, Y_test))
        score_list.append(svm.score(H_test, Y_test))
        # 填充score列表， 用于输出分类准确度（这里要在测试集上， 千万不能在训练集上测试了）
        print("score = ", svm.score(H_test, Y_test))
        # 这些都是调试信息
        svm_info_temp = SVM_info(svm.support_, svm.n_support_, svm.dual_coef_)
        SVM_info_list.append(svm_info_temp)
        
    return score_list, arg_C_list, SVM_info_list

def store_lists(args_list, score_list):
    # 下面几行是把list存下来（存在.pkl文件里面）
    with open(f'./pickles/{mode}_score_list', 'wb') as file_handle:
        pickle.dump(score_list, file_handle)
    with open(f'./pickles/{mode}_args_list', 'wb') as file_handle:
        pickle.dump(args_list, file_handle)

def plot_C(args_list, score_list, mode):
    # 使用 matplotlib 画图
    plt.figure(figsize=(10, 6))
    # 使用采样点作为横坐标， 不然图不好看
    plt.plot(range(len(args_list)), score_list, marker='o', linestyle='-')

    plt.xticks(range(len(args_list)), ['{:.2f}'.format(c) for c in args_list])  
    plt.title(f'{mode} SVC Accuracy vs. Regularization Parameter C') 
    plt.xlabel('C parameter value')
    plt.ylabel('Accuracy score') 

    # 设置横坐标显示到小数点后三位
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # 旋转刻度， 使其更易读
    plt.xticks(rotation=45)
    # 显示网格线
    plt.grid(True)
    plt.show()

def plot_sv_num(args_list, SVM_info_list, mode):
    support_vectors_counts = []

    for svm_info in SVM_info_list:
        dual_coefs_times_label = svm_info.dual_coefs_times_label

        # 计算每个模型中alpha > 0的支持向量数量
        # 这里我们假设二分类问题，dual_coef_是一个一行的2D数组
        print("dual_coefs : " , dual_coefs_times_label[0])
        support_vectors_count = sum(coef > 0 for coef in dual_coefs_times_label[0])
        
        support_vectors_counts.append(support_vectors_count)

    plt.figure(figsize=(10, 6))

    print("args_list : ", args_list)
    print("support_vectors_counts : ", support_vectors_counts)

    plt.plot(range(len(args_list)), support_vectors_counts, marker='o', linestyle='-')

    plt.xticks(range(len(args_list)), [f'{c:.2f}' for c in args_list], rotation=45)

    plt.title(f'Number of Support Vectors vs. Regularization Parameter C (kernel: {mode})')
    plt.xlabel('C parameter value')
    plt.ylabel('Number of Support Vectors')
    plt.grid(True)
    plt.savefig('./pickles/sv_num.png')
    plt.show()

if __name__ == '__main__':
######################## Get train/test dataset ########################
    X_train,X_test,Y_train,Y_test = get_data('dataset')
########################## Get HoG featues #############################
    H_train,H_test = get_HOG(X_train), get_HOG(X_test)
######################## standardize the HoG features ####################
    H_train,H_test = standardize(H_train), standardize(H_test)
#######################################################################
####################### Implement you code here #######################
#######################################################################

    mode = 'linear'
    #每次只需要改这个mode就行了， 其他地方都会自动替换
    
    score_list, args_list, SVM_info_list = training(mode, H_train, Y_train, H_test, Y_test)
    # 训练+测试的主要过程

    print('args : ', args_list)
    print('score : ', score_list)

    store_lists(args_list, score_list)

    #plot_C(args_list, score_list, mode)
    plot_sv_num(args_list, SVM_info_list, mode)
    
