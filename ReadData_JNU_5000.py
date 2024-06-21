import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
import random
import pandas as pd


# 用训练集标准差标准化训练集以及测试集
def scalar_stand(data_x):
    # data_x = data_x.reshape(-1, 1)
    scalar = preprocessing.StandardScaler().fit(data_x)
    data_x = scalar.transform(data_x)
    data_x = data_x.reshape(-1)
    return data_x


# 构建文件读取函数capture,返回原始数据和标签数据
def capture(original_path):  # 读取mat文件，返回一个属性的字典
    filenames = os.listdir(original_path)  # 得到负载文件夹下面的10个文件名
    Data_use = []
    for i in filenames:  # 遍历10个文件数据
        # 文件路径
        file_path = os.path.join(original_path,
                                 i)  # 选定一个数据文件的路径
        file = pd.read_csv(
            file_path)  # 数组
        Data_use.append(file)
    return Data_use


# 划分训练样点集和测试样点集
def spilt(data, rate):  # [[N1],[N2],...,[N10]]
    tra_data = []
    te_data = []
    val_data = []
    for i in range(len(data)):  # 遍历所有文件夹
        slice_data = data[i]  # 选取1个文件中的数据
        slice_data = scalar_stand(slice_data)
        all_length = len(slice_data)  # 文件中的数据长度
        # print('数据总数为',i,  all_length)
        tra = np.array(slice_data[0:int(all_length * rate[0])]).flatten()
        tra_data.append(tra)
        # print("训练样本点数", len(tra_data[i]))
        val = np.array(slice_data[int(all_length * rate[0]):int(all_length * (rate[0] + rate[1]))]).flatten()
        val_data.append(val)
        tes = np.array(slice_data[int(all_length * (rate[0] + rate[1])):]).flatten()
        te_data.append(tes)
        # print("测试样本点数", len(te_data[i]))
        # 行列转换

    return tra_data, val_data, te_data


def sampling(data, stride, sample_len):
    sample = []
    label = []
    for i in range(len(
            data)):  # 遍历10个文件
        all_length = len(
            data[i])  # 文件中的数据长度
        # print('采样的训练数据总数为', all_length)
        number_sample = int(
            (all_length - sample_len) / stride + 1)  # 样本数
        # print('采样的训练数据总数为', i)
        # print("number=", number_sample)
        for j in range(
                number_sample):  # 逐个采样
            sample.append(data[i][j * stride: j * stride + sample_len])
            label.append(i)
            j += 1
    return sample, label


def get_data(path, rate, stride, sample_len):
    data = capture(path)  # 读取数据
    train_data, val_data, test_data = spilt(data, rate)  # 列表[N1,N2,N10]
    x_train, y_train = sampling(train_data, stride, sample_len)
    x_validate, y_validate = sampling(val_data, stride, sample_len)
    x_test, y_test = sampling(test_data, stride, sample_len)
    return np.array(x_train), np.array(y_train), np.array(x_validate), np.array(y_validate), np.array(x_test), np.array(
        y_test)



path = r'data_jnu/1000'
sample_len = 500
stride = int(sample_len/2)
rate = [1.0, 0, 0]
x_train, y_train, x_validate, y_validate, x_test, y_test = get_data(path, rate, stride, sample_len)
print(x_train.shape)  # (2677, 400, 2)
print(x_validate.shape) # (330, 400, 2)
print(x_test.shape)  # (330, 400, 2)
print(y_train.shape)