import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pre_processing import FILE_NAME

N, S, V, F = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
ecg_data = {'N': N, 'S': S, 'V': V, 'F': F}
categorys = ('N', 'S', 'V', 'F')

# 筛选四种类别的心拍
for file_name in FILE_NAME:
    outcome = pd.read_csv(f'data/pre_processed_data/{file_name}_outcome.csv')
    data = pd.read_csv(f'data/pre_processed_data/{file_name}_data.csv',
                       header=None)
    for category in categorys:
        ecg_data[category] = ecg_data[category].append(
            data[outcome['symbol'] == category], ignore_index=True)

# 绘制四种类别的心拍
# 设置字体大小
plt.rcParams['font.size'] = 10

for i, category in enumerate(categorys):
    # 随机选择100个心拍进行绘制
    heartbeats = ecg_data[category].sample(n=100,
                                           random_state=100,
                                           ignore_index=True)

    # viridis colormap自动分配颜色
    colors = plt.cm.get_cmap('viridis', 90)

    # 绘制2*2张子图
    plt.subplot(2, 2, i + 1)
    plt.title(f'{category} Heartbeat')
    plt.xlabel('Time (1/360s)')
    plt.ylabel('Voltage (mV)')
    for i, row in heartbeats.iterrows():
        plt.plot(row, color=colors(i), linestyle='-',
                 marker=None)  # 移除marker参数以不显示点
    plt.tight_layout()  # 自动调整子图间距

plt.savefig('figures/four_heartbeat.jpg', dpi=300)  # 保存图片
