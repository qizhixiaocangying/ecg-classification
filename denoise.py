import pywt
import numpy as np
import scipy.signal as signal
from math import sqrt
from matplotlib import pyplot as plt
from read_data import read_signal


# 使用IIR陷波器滤波器滤除ECG信号中的60Hz干扰
def ecg_iirnotch_60hz(ecg_signal, fs, Q=60):
    # 设计IIR陷波器滤波器
    b, a = signal.iirnotch(60, Q, fs)

    # 对ECG信号进行滤波处理
    ecg_signal_filtered = signal.lfilter(b, a, ecg_signal)

    return ecg_signal_filtered


# 定义一个函数，用于对心电图信号进行中值滤波
def ecg_medfilt(ecg_signal, window_size):
    # 确保窗口大小为奇数
    window_size = window_size + (1 - window_size % 2)
    pad_size = window_size // 2  # 计算需要扩展的大小

    # 扩展信号
    padded_signal = np.concatenate((
        ecg_signal[:pad_size][::-1],  # 镜像信号的开始部分  
        ecg_signal,
        ecg_signal[-pad_size:][::-1]  # 镜像信号的结束部分  
    ))

    # 对扩展后的信号进行中值滤波
    ecg_baseline = signal.medfilt(padded_signal, window_size)

    # 从滤波后的信号中提取与原始信号长度相同的部分
    ecg_baseline_filtered = ecg_baseline[pad_size:-pad_size]

    # 计算并添加总体偏差
    totality_bias = np.mean(ecg_baseline_filtered)
    ecg_signal_filtered = ecg_signal - (ecg_baseline_filtered - totality_bias)

    return ecg_signal_filtered


# 小波阈值去噪算法
def ecg_wavelet_denoise(ecg_signal, wavelet_name='sym4', level=8):
    # 对心电信号进行小波分解
    coeffs = pywt.wavedec(ecg_signal, wavelet_name, level=level)

    # 计算阈值
    sigma = (np.median(np.abs(coeffs[-1])) / 0.6745)
    threshold = sigma * np.sqrt(2 * np.log(len(ecg_signal)))

    # 对细节系数进行阈值处理
    c = sqrt(2) / 2
    for i in range(1, level + 1):
        coeffs[-i] = _improved_threshold_function(coeffs[-i], c**i * threshold)
    # 对近似系数进行阈值处理
    coeffs[0] = _improved_threshold_function(coeffs[0], c**level * threshold)

    # 对信号进行小波重构
    ecg_signal_filtered = pywt.waverec(coeffs, wavelet_name)

    return ecg_signal_filtered


# 改进的小波阈值函数
def _improved_threshold_function(data, threshold, alpha=5):
    return np.where(
        np.abs(data) >= threshold,
        data * np.tanh(alpha * (np.abs(data) - threshold)), 0)


# 去除ECG信号中的噪声
def ecg_denoise(ecg_signal, fs):
    # 对ECG信号进行IIR陷波器滤波，滤除60Hz工频干扰
    ecg_signal_filtered = ecg_iirnotch_60hz(ecg_signal, fs)

    # 对滤波后的信号进行中值滤波，去除基线漂移
    ecg_signal_filtered = ecg_medfilt(ecg_signal_filtered, int(0.8 * fs))

    # 对滤波后的信号进行小波阈值去噪，去除肌电干扰
    ecg_signal_filtered = ecg_wavelet_denoise(ecg_signal_filtered)

    return ecg_signal_filtered


# 绘制去噪前后的ECG信号
def _plot_ecg_signal(ecg_signal, ecg_signal_filtered, title):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(ecg_signal, label='Original ECG')
    plt.ylabel('Voltage (mV)')
    plt.legend(loc='upper left')

    plt.subplot(2, 1, 2)
    plt.plot(ecg_signal_filtered, label='Filtered ECG')
    plt.xlabel('Time (1/360s)')
    plt.ylabel('Voltage (mV)')
    plt.legend(loc='upper left')
    plt.suptitle(title)
    plt.savefig(f'figures/{title}.jpg', dpi=300)
    plt.show()


if __name__ == '__main__':
    ecg_original_signal, fs = read_signal('data/MIT-BIH/100')
    ecg_signal = ecg_original_signal[0:10000]
    ecg_signal_filtered = ecg_denoise(ecg_signal, fs)
    _plot_ecg_signal(ecg_signal, ecg_signal_filtered, 'ECG Denoise')
