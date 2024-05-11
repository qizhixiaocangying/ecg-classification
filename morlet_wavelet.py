import numpy as np
import matplotlib.pyplot as plt


# Morlet小波函数
def morlet_wavelet(t, s=1.0, w=5.0):
    return np.exp(-t**2 / (2 * s**2)) * np.cos(w * t)


# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 时间范围和采样点
t = np.linspace(-5, 5, 1000)
wavelet = morlet_wavelet(t)

# 绘制图像
plt.figure(figsize=(8, 4))
plt.plot(t, wavelet, label="Morlet Wavelet")
# plt.title("Morlet小波函数示意图")
plt.xlabel("时间")
plt.ylabel("振幅")
plt.legend()
plt.grid(True)
plt.savefig("figures/morlet_wavelet.jpg", dpi=300)
