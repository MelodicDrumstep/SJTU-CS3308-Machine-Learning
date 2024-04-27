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
from sklearn.manifold import TSNE

# 用于保存 SVM 模型的相关信息的一个类
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

# 每次批量绘图前先用这个测试一下
def test_fill(arg_C_list):
    arg_C_list.append(1)

# 用于训练模型并生成score列表的函数
def training(mode, H_train, Y_train, H_test, Y_test):
    arg_C_list = []
    fill_arg_C_list(0.001, 0.01, 0.005, arg_C_list)
    fill_arg_C_list(0.01, 0.1, 0.05, arg_C_list)
    fill_arg_C_list(0.1, 1, 0.5, arg_C_list)
    fill_arg_C_list(1, 10, 5, arg_C_list)
    fill_arg_gamma_list(arg_C_list)
    #test_fill(arg_C_list)
    # 填充参数列表 arg_C_list

    score_list = []
    SVM_info_list = []
    indexed_scores_list = []
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
        #experiment(svm_info_temp, argC)
        # 获取决策函数的值
        decision_scores = svm.decision_function(H_train)
        # 结合分类的标签和决策函数，创建一个复合的列表
        indexed_scores = list(zip(range(len(Y_train)), Y_train, decision_scores))
        indexed_scores_list.append(indexed_scores)
        
    return score_list, arg_C_list, SVM_info_list, indexed_scores_list

# 把list存起来复用的函数
def store_lists(args_list, score_list):
    # 下面几行是把list存下来（存在.pkl文件里面）
    with open(f'./pickles/{mode}_score_list', 'wb') as file_handle:
        pickle.dump(score_list, file_handle)
    with open(f'./pickles/{mode}_args_list', 'wb') as file_handle:
        pickle.dump(args_list, file_handle)

# 绘制基础任务的函数， 即分类准确度和 C 的关系
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

# 一个实验函数， 看输出性质
def experiment(SVM_info, C):
    # 用于调试的函数
    print("C : ", C)
    print("support_vector_indices_length : ", len(SVM_info.support_vector_indices))
    print("support_vector_indices : ", SVM_info.support_vector_indices)
    print("num_support_vectors_each_class[0] : ", SVM_info.num_support_vectors_each_class[0])
    print("num_support_vectors_each_class[1] : ", SVM_info.num_support_vectors_each_class[1])
    print("dual_coefs_times_label_length : ", len(SVM_info.dual_coefs_times_label))
    print("dual_coefs_times_label[0] : ", SVM_info.dual_coefs_times_label[0])
    sum_coef_C = 0
    for coef in SVM_info.dual_coefs_times_label[0]:
        #print("coef : ", coef)
        if(abs(abs(coef) - C) < 1e-5):
            sum_coef_C += 1
    print("sum_coef_C : ", sum_coef_C)

# 绘制支持向量个数图的函数
def plot_sv_num(args_list, SVM_info_list, mode):
    print("num of SVM_info_list : ", len(SVM_info_list))
    support_vectors_counts = []
    for svm_info in SVM_info_list:
        print("num of support_vectors_each_class[0] : ", svm_info.num_support_vectors_each_class[0])
        print("num of support_vectors_each_class[1] : ", svm_info.num_support_vectors_each_class[1])
        support_vectors_counts.append(svm_info.num_support_vectors_each_class[0] + svm_info.num_support_vectors_each_class[1])

    plt.figure(figsize=(10, 6))

    print("args_list : ", args_list)
    print("support_vectors_counts : ", support_vectors_counts)
    # 启用latex
    plt.rc('text', usetex=True)

    plt.plot(range(len(args_list)), support_vectors_counts, marker='o', linestyle='-')

    plt.xticks(range(len(args_list)), [f'{c:.2f}' for c in args_list], rotation=45)

    plt.title(rf'Number of $\alpha_i > 0$ vectors  vs. Regularization Parameter C (kernel: {mode})')
    plt.xlabel('C parameter value')
    plt.ylabel(r'Number of $\alpha_i > 0$ vectors')
    plt.grid(True)
    plt.savefig(f'./pictures/sv_num_{mode}.png')
    plt.show()

# 绘制分类信心最大的5个样本的原图像函数
def plot_confidence_vectors_origin_picture(X_train, args_list, SVM_info_list, indexed_scores_list, mode):
    for i in range(0, len(args_list)):
        plt.figure(figsize=(10, 6))
        argC = args_list[i]
        indexed_scores = indexed_scores_list[i]
        print("indexed_scores :", indexed_scores)
        top_positive_samples = sorted([s for s in indexed_scores if s[1] == 3], key=lambda x: -x[2])[:5]
        top_negative_samples = sorted([s for s in indexed_scores if s[1] == 2], key=lambda x: x[2])[:5]
        print("top_positve_samples :", top_positive_samples)

        plot_confidence_origin_helper(X_train, top_positive_samples, "positive", argC, mode)
        plot_confidence_origin_helper(X_train, top_negative_samples, "negative", argC, mode)

# 上一个函数的辅助函数， 实现接口封装
def plot_confidence_origin_helper(X_train, top_samples, label, argC, mode):
    print("label : ", label)
    print("argC: " ,argC)
    indexes = [item[0] for item in top_samples]
    print("indexes : ", indexes)
    plt.figure(figsize=(15, 3))
    plt.suptitle(f'Top 5 {label} Samples with Highest Confidence, C = {argC}, kernel: {mode}', fontsize=20)
    for i in range(0, len(top_samples)):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X_train[indexes[i]], cmap='gray')
        plt.title(f"Sample {indexes[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'./pictures/confidence/confidence_origin_{mode}.png')
    #plt.show()

# 绘制分类信心最大的5个样本的决策函数值的函数
def plot_confidence_vectors(args_list, SVM_info_list, indexed_scores_list, mode):
    for i in range(0, len(args_list)):
        plt.figure(figsize=(10, 6))
        argC = args_list[i]
        indexed_scores = indexed_scores_list[i]
        print("indexed_scores :", indexed_scores)
        top_positive_samples = sorted([s for s in indexed_scores if s[1] == 3], key=lambda x: -x[2])[:5]
        top_negative_samples = sorted([s for s in indexed_scores if s[1] == 2], key=lambda x: x[2])[:5]
        print("top_positve_samples :", top_positive_samples)

        plot_confidence_helper(top_positive_samples, "positive", argC, mode)
        plot_confidence_helper(top_negative_samples, "negative", argC, mode)

# 上一个函数的辅助函数， 实现接口封装
def plot_confidence_helper(top_samples, label, argC, mode):
    print("label : ", label)
    print("argC: " ,argC)
    plt.figure(figsize=(10, 6))
    indexes = [item[0] for item in top_samples]
    values = [item[2] for item in top_samples]
    print("indexes : ", indexes)
    print("values : " , values)
    plt.plot(range(len(indexes)), values, marker='o', linestyle='-')

    plt.title(rf'The top 5 {label} samples that we can classify with confidence, C = {argC}, kernel : {mode}')
    plt.xlabel('the 5 samples')
    plt.ylabel(r'The decision value of the sample')
    plt.grid(True)
    plt.savefig(f'./pictures/confidence/confidence_{label}_{argC}.png')
    plt.show()

# 绘制5个支持向量样本图像的函数
def plot_N_sv_sample(X_train, args_list, SVM_info_list, N, mode):
    for i in range(0, len(args_list)):
        argC = args_list[i]
        svm_info = SVM_info_list[i]
        support_vector_indices = svm_info.support_vector_indices
        sv_list = []
        for i in range(0, N):
            sv_list.append(support_vector_indices[i])

        print("sv_list : ", sv_list)
        plt.figure(figsize=(15, 3))
        plt.suptitle(f'5 Support Vector Sample, C = {argC}, kernel: {mode}', fontsize=20)
        for i in range(0, N):
            plt.subplot(1, 5, i + 1)
            plt.imshow(X_train[sv_list[i]], cmap='gray')
            plt.title(f"Sample {sv_list[i]}")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'./pictures/sv/sv_{argC}_{mode}.png')
        #pplt.show()


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
    score_list, args_list, SVM_info_list, indexed_scores_list = training(mode, H_train, Y_train, H_test, Y_test)
    # 训练+测试的主要过程

    print('args : ', args_list)
    print('score : ', score_list)

    store_lists(args_list, score_list)

    # 以下函数都是画图函数， 依次调试好执行
    #plot_C(args_list, score_list, mode)
    #plot_sv_num(args_list, SVM_info_list, mode)
    #plot_confidence_vectors(args_list, SVM_info_list, indexed_scores_list, mode)
    #plot_confidence_vectors_origin_picture(X_train, args_list, SVM_info_list, indexed_scores_list, mode)
    plot_N_sv_sample(X_train, args_list, SVM_info_list, 5, mode)
