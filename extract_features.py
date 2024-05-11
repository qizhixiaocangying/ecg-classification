import numpy as np
import pandas as pd
import pywt as wt
import scipy.signal as signal
import scipy.stats as stats
import neurokit2 as nk
from PyEMD import EEMD
from pre_processing import FILE_NAME
from read_data import read_age_sex


# 提取时域特征
def extract_time_domain_features(r_peaks_indices):
    features_rr = pd.DataFrame()

    # pre_r: 前一个rr间距
    # post_r: 后一个rr间距
    # local_r: 前十个pre_r的平均值
    # global_r: 最近20分钟内pre_r的平均值
    pre_r = np.array([])
    post_r = np.array([])
    local_r = np.array([])
    global_r = np.array([])

    # pre_r 和 post_r
    pre_r = np.append(pre_r, 0)
    post_r = np.append(post_r, r_peaks_indices[1] - r_peaks_indices[0])

    for i in range(1, len(r_peaks_indices) - 1):
        pre_r = np.append(pre_r, r_peaks_indices[i] - r_peaks_indices[i - 1])
        post_r = np.append(post_r, r_peaks_indices[i + 1] - r_peaks_indices[i])

    pre_r[0] = pre_r[1]
    pre_r = np.append(pre_r, r_peaks_indices[-1] - r_peaks_indices[-2])
    post_r = np.append(post_r, post_r[-1])

    # local_r
    for i in range(0, len(r_peaks_indices)):
        num = 0
        avg_val = 0
        for j in range(-9, 1):
            if j + i >= 0:
                avg_val += pre_r[i + j]
                num += 1
        local_r = np.append(local_r, avg_val / num)

    # global_r, 20 * 60 * 360 = 432000 个采样点
    global_r = np.append(global_r, pre_r[0])
    for i in range(1, len(r_peaks_indices)):
        num = 0
        avg_val = 0

        for j in range(0, i + 1):
            if (r_peaks_indices[i] - r_peaks_indices[j]) < 432000:
                avg_val += pre_r[j]
                num += 1
        global_r = np.append(global_r, avg_val / num)

    # 所有RR间期的平均值
    avg_r = np.mean(np.diff(r_peaks_indices))

    # 归一化的pre_r，post_r，local_r，global_r
    norm_pre_r = pre_r / avg_r
    norm_post_r = post_r / avg_r
    norm_local_r = local_r / avg_r
    norm_global_r = global_r / avg_r

    # 特征值
    features_rr['pre_r'] = pre_r
    features_rr['post_r'] = post_r
    features_rr['local_r'] = local_r
    features_rr['global_r'] = global_r
    features_rr['norm_pre_r'] = norm_pre_r
    features_rr['norm_post_r'] = norm_post_r
    features_rr['norm_local_r'] = norm_local_r
    features_rr['norm_global_r'] = norm_global_r

    return features_rr


# 提取频域特征
def extract_frequency_domain_features(heart_beats, fs):
    features_fd = pd.DataFrame()

    # ps_exp: 功率谱的期望
    # ps_std: 功率谱的标准差
    # ps_skew: 功率谱的偏度
    # ps_kurt: 功率谱的峰度
    # freq_exp: 频率期望
    # freq_std: 频率标准差
    # freq_peak: 频率峰值
    # freq_bw: 频率带宽
    ps_exp = np.zeros(len(heart_beats))
    ps_std = np.zeros(len(heart_beats))
    ps_skew = np.zeros(len(heart_beats))
    ps_kurt = np.zeros(len(heart_beats))
    freq_exp = np.zeros(len(heart_beats))
    freq_std = np.zeros(len(heart_beats))
    freq_peak = np.zeros(len(heart_beats))
    freq_bw = np.zeros(len(heart_beats))

    for i in range(len(heart_beats)):
        # 计算信号的功率谱
        freqs, ps = signal.welch(heart_beats.iloc[i],
                                 fs,
                                 scaling='spectrum',
                                 nperseg=100,
                                 noverlap=50)

        # 功率谱的期望
        ps_exp[i] = np.mean(ps)

        # 功率谱的标准差
        ps_std[i] = np.std(ps)

        # 功率谱的偏度
        ps_skew[i] = stats.skew(ps)

        # 功率谱的峰度
        ps_kurt[i] = stats.kurtosis(ps, fisher=True)

        # 频率期望
        freq_exp[i] = np.sum(freqs * ps) / np.sum(ps)

        # 频率标准差
        freq_std[i] = np.sqrt(
            np.sum((freqs - freq_exp[i])**2 * ps) / np.sum(ps))

        # 频率峰值
        freq_peak[i] = freqs[np.argmax(ps)]

        # 频率带宽
        freq_bw[i] = np.abs(freqs[np.argmax(ps)] - freqs[np.argmin(ps)])

    # 特征值
    features_fd['ps_exp'] = ps_exp
    features_fd['ps_std'] = ps_std
    features_fd['ps_skew'] = ps_skew
    features_fd['ps_kurt'] = ps_kurt
    features_fd['freq_exp'] = freq_exp
    features_fd['freq_std'] = freq_std
    features_fd['freq_peak'] = freq_peak
    features_fd['freq_bw'] = freq_bw

    return features_fd


# 提取时频域特征
def extract_time_frequency_features(heart_beats, fs):
    # 设置STFT参数
    window_type = 'hann'  # 窗口类型
    nperseg = 64  # 窗口大小
    noverlap = nperseg // 2  # 重叠长度
    nfft = nperseg  # FFT点数

    # 设置小波变换参数
    wavelet_type = 'db1'  # 小波类型
    level = 3  # 小波层数

    # 初始化特征列表
    mean_features = []
    std_features = []
    cA3_features = []

    for i in range(len(heart_beats)):
        heart_beat = heart_beats.iloc[i]
        # 短时傅里叶变换
        _, _, stft = signal.stft(heart_beat,
                                 fs,
                                 window=window_type,
                                 nperseg=nperseg,
                                 noverlap=noverlap,
                                 nfft=nfft)

        # 提取时频统计特征
        # 计算每个时间窗口的频谱的均值和标准差
        mean_spectrum = np.mean(np.abs(stft), axis=0)
        std_spectrum = np.std(np.abs(stft), axis=0)

        # 小波变换
        coeffs = wt.wavedec(heart_beat, wavelet_type, level=level)
        cA3 = coeffs[0]

        # 将特征添加到特征列表中
        mean_features.append(mean_spectrum)
        std_features.append(std_spectrum)
        cA3_features.append(cA3)

    # 特征值
    mean_columns = [f'mean_freq_{i + 1}' for i in range(len(mean_spectrum))]
    std_columns = [f'std_freq_{i + 1}' for i in range(len(std_spectrum))]
    cA3_columns = [f'cA3_wavelet_{i + 1}' for i in range(len(cA3))]
    features_tf = pd.DataFrame(
        np.concatenate((mean_features, std_features, cA3_features), axis=1),
        columns=mean_columns + std_columns + cA3_columns)

    return features_tf


# 提取非线性特征
def extract_nonlinear_features(heart_beats):
    features_nl = pd.DataFrame()

    # apen: 近似熵
    # sampen: 样本熵
    # fuzzyen: 模糊熵
    apen = np.zeros(len(heart_beats))
    sampen = np.zeros(len(heart_beats))
    fuzzyen = np.zeros(len(heart_beats))

    for i in range(len(heart_beats)):
        # 近似熵
        apeni, _ = nk.entropy_approximate(heart_beats.iloc[i])
        apen[i] = apeni

        # 样本熵
        sampeni, _ = nk.entropy_sample(heart_beats.iloc[i])
        sampen[i] = sampeni

        # 模糊熵
        fuzzyeni, _ = nk.entropy_fuzzy(heart_beats.iloc[i])
        fuzzyen[i] = fuzzyeni

    # 特征值
    features_nl['apen'] = apen
    features_nl['sampen'] = sampen
    features_nl['fuzzyen'] = fuzzyen

    return features_nl


# 提取分解域特征
def extract_decomposition_features(heart_beats):
    # 设置嵌入维数
    emb_dim = 5

    # 设置EEMD参数
    N = 30  # 总体平均次数 N
    r = 0.1  # 白噪声宽度 r

    # 设置Welch参数
    fs = 360  # 采样频率
    nperseg = 50  # 窗长度
    nfft = 256  # FFT点数
    noverlap = nperseg // 2  # 窗口重叠率为50%

    # 初始化特征列表
    sv_features = []
    kimfs_time_features = []
    kimfs_freq_features = []

    for i in range(len(heart_beats)):
        heart_beat = np.array(heart_beats.iloc[i])

        # 创建EEMD实例并设置参数
        eemd_inst = EEMD()
        eemd_inst.trials = N
        eemd_inst.noise_width = r

        # 进行EEMD分解
        imfs = eemd_inst.eemd(heart_beat)
        kimfs = imfs[:4]

        # EEMD特征
        imf_time_features = []
        imf_freq_features = []
        for imf in kimfs:
            mean = np.mean(imf)  # 平均值
            std = np.std(imf)  # 标准差
            max_val = np.max(imf)  # 最大值
            min_val = np.min(imf)  # 最小值
            zcr = np.mean(np.diff(np.sign(imf)) != 0)  # 零交叉率
            imf_time_features.extend([mean, std, max_val, min_val, zcr])

            # 频域特征
            f, Pxx = signal.welch(imf,
                                  fs,
                                  'boxcar',
                                  nperseg=nperseg,
                                  nfft=nfft,
                                  noverlap=noverlap)
            total_energy = np.sum(Pxx)  # 总能量
            major_freq = f[np.argmax(Pxx)]  # 主频率
            band_energy = np.sum(Pxx[(f >= 1) & (f <= 50)])  # 1~50Hz频带能量
            imf_freq_features.extend([total_energy, major_freq, band_energy])

        # 重构相空间
        windows = np.lib.stride_tricks.sliding_window_view(heart_beat, emb_dim)

        # 进行奇异值分解
        U, S, Vh = np.linalg.svd(windows, full_matrices=False)

        # 将特征添加到特征列表中
        sv_features.append(S)
        kimfs_time_features.append(imf_time_features)
        kimfs_freq_features.append(imf_freq_features)

    # 特征名称
    sv_columns = [f'sv_{i + 1}' for i in range(len(S))]
    time_columns = ['mean', 'std', 'max_val', 'min_val', 'zcr']
    freq_columns = ['total_energy', 'major_freq', 'band_energy']
    kimfs_time_columns = [
        f'kimfs_time_{i + 1}_{time_columns[j]}' for i in range(4)
        for j in range(5)
    ]
    kimfs_freq_columns = [
        f'kimfs_freq_{i + 1}_{freq_columns[j]}' for i in range(4)
        for j in range(3)
    ]

    # 特征值
    features_dc = pd.DataFrame(
        np.concatenate((sv_features, kimfs_time_features, kimfs_freq_features),
                       axis=1),
        columns=sv_columns + kimfs_time_columns + kimfs_freq_columns)

    return features_dc


# 提取宏观特征
def extract_macro_features(heart_beats, file_name):
    features_mc = pd.DataFrame()

    # 读取年龄和性别
    age, sex = read_age_sex(file_name)

    # 如果年龄未记录，则用平均值代替
    if age == -1:
        age = 63

    # 特征值
    features_mc['age'] = np.full(len(heart_beats), age)
    features_mc['sex'] = np.full(len(heart_beats), sex)

    return features_mc


# 划分训练集和测试集
def split_train_test():
    # 训练集
    DS1 = [
        '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
        '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
        '223', '230'
    ]

    # 测试集
    DS2 = [
        '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
        '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
        '233', '234'
    ]

    train = pd.DataFrame()
    test = pd.DataFrame()

    # 划分训练集
    for f_name in DS1:
        # 读取特征数据和标签
        features = pd.read_csv(f'data/features/{f_name}_features.csv')
        outcome = pd.read_csv(f'data/pre_processed_data/{f_name}_outcome.csv')

        # 合并特征和标签
        data = pd.concat([features, outcome[['symbol']]], axis=1)

        # 加入训练集
        train = pd.concat([train, data])

    # 打乱训练集顺序
    train = train.sample(frac=1).reset_index(drop=True)

    # 划分测试集
    for f_name in DS2:
        # 读取特征数据和标签
        features = pd.read_csv(f'data/features/{f_name}_features.csv')
        outcome = pd.read_csv(f'data/pre_processed_data/{f_name}_outcome.csv')

        # 合并特征和标签
        data = pd.concat([features, outcome[['symbol']]], axis=1)

        # 加入测试集
        test = pd.concat([test, data])

    # 保存训练集和测试集
    train.to_csv('data/features/train_data.csv', index=False)
    test.to_csv('data/features/test_data.csv', index=False)


if __name__ == '__main__':
    for f_name in FILE_NAME:
        # 读取心拍数据和标签
        data = pd.read_csv(f'data/pre_processed_data/{f_name}_data.csv',
                           header=None)
        outcome = pd.read_csv(f'data/pre_processed_data/{f_name}_outcome.csv')

        # 提取时域特征
        features_rr = extract_time_domain_features(np.array(outcome['sample']))

        # 提取频域特征
        features_fd = extract_frequency_domain_features(data, 360)

        # 提取时频域特征
        features_tf = extract_time_frequency_features(data, 360)

        # 提取非线性特征
        features_nl = extract_nonlinear_features(data)

        # 提取分解域特征
        features_dc = extract_decomposition_features(data)

        # 提取宏观特征
        file_name = f'data/MIT-BIH/{f_name}'
        features_mc = extract_macro_features(data, file_name)

        # 合并特征并保存
        features = pd.concat([
            features_rr, features_fd, features_tf, features_nl, features_dc,
            features_mc
        ],
                             axis=1)
        features.to_csv(f'data/features/{f_name}_features.csv', index=False)

    # 划分训练集和测试集
    split_train_test()
