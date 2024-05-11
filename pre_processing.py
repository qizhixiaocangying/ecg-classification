import csv
import numpy as np
import denoise
from read_data import *

ECG_R_LIST = [
    'N', 'f', 'e', '/', 'j', 'L', 'R', 'S', 'A', 'J', 'a', 'V', 'E', 'F', 'Q'
]
AAMI_MIT = {
    'N': 'NLRej',  # 将15类信号分为五大类
    'S': 'AaJS',
    'V': 'VE',
    'F': 'F',
    'Q': '/fQ'
}
AAMI_MIT_INV = {s: k for k, v in AAMI_MIT.items() for s in v}
FILE_NAME = [
    '100', '101', '103', '105', '106', '108', '109', '111', '112', '113',
    '114', '115', '116', '117', '118', '119', '121', '122', '123', '124',
    '200', '201', '202', '203', '205', '207', '208', '209', '210', '212',
    '213', '214', '215', '219', '220', '221', '222', '223', '228', '230',
    '231', '232', '233', '234'
]


# 提取R点的标签和坐标
def get_R_symbol_sample(annotation):
    # 获取表示R点的心拍类型的索引
    index = np.isin(annotation.symbol, ECG_R_LIST)

    # 提取表示为R点的心拍标签
    symbol = np.array(annotation.symbol)[index]
    symbol = np.array([AAMI_MIT_INV[s] for s in symbol])

    # 提取表示为R点的坐标
    sample = np.array(annotation.sample)[index]
    return symbol, sample


# 心拍分割
def split_heartbeat(signal, annotation):
    # 获取R点的标签和坐标
    symbol, sample = get_R_symbol_sample(annotation)

    length = len(signal)
    result_x = []
    result_y = [['sample', 'symbol']]
    # 遍历每一个R点的坐标，截取心拍信号
    for list_index, signal_index in enumerate(sample):
        # 前后分别截取 100 和 100 个样本点共 200 个样本点作为样本心拍
        if 100 < signal_index < length - 100:
            heartbeat = signal[signal_index - 100:signal_index + 100]
            result_x.append(heartbeat)
            result_y.append([signal_index, symbol[list_index]])
    return result_x, result_y


# 保存到CSV格式的文件
def save_csv(file_name, data):
    with open(f'data/pre_processed_data/{file_name}.csv', 'w',
              newline='\n') as f:
        writer = csv.writer(f)
        # 将列表的每条数据依次写入csv文件， 并以逗号分隔
        # 传入的数据为列表中嵌套列表，每一个列表为每一行的数据
        writer.writerows(data)


# 主函数
if __name__ == '__main__':
    for f_name in FILE_NAME:
        file_name = f'data/MIT-BIH/{f_name}'

        # 读取信号，标签
        signal, fs = read_signal(file_name)
        annotation = read_annotation(file_name)

        # 去除信号中的噪声
        signal = denoise.ecg_denoise(signal, fs)

        # 心拍分割
        result_x, result_y = split_heartbeat(signal, annotation)

        # 保存到CSV文件
        save_csv(f_name + '_data', result_x)
        save_csv(f_name + '_outcome', result_y)
