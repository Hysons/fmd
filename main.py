import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin,lfilter,correlate,find_peaks
import pandas as pd

#数据导入
rows = 167
#从0到1的167个等间隔的浮点数
time = np.linspace(0,1,rows)
signal_data = pd.read_csv('data\data.csv')
#signal_df = signal_data
signal_df = pd.DataFrame(signal_data)
signal_df.columns=['Signal']
#print(signal_df)

#汉宁窗口初始化FIR滤波器组
def initialize_filters(L,k):
    filters = []
    for k in range(1,k+1):
        cutoff = 0.5 / k
        filter = firwin(L,cutoff,window='hann')
        filters.append(filter)
    return filters

#自相关普,用于周期性信号的周期估计
def estimate_period(signal):
    correlation = correlate(signal,signal,mode='full')
    correlation = correlation[len(correlation)//2:]
    peaks,_ = find_peaks(correlation)
    #如果找到的峰值数量大于1，那么周期被认为是第二个峰值的索引（peaks[1]），因为第一个峰值通常是在0的位置，对应于信号与自身的完全重叠。
    if len(peaks)>1:
        period = peaks[1]
    else:
        period = len(signal)
    return period

#fmd函数
def fmd(signal,n,L=100,max_iters=10):
    k = min(10, max(5,n))
    filters = initialize_filters(L,k)
    modes = []
    signal = signal.values.flatten() if isinstance(signal,pd.DataFrame) else signal.flatten()

    for i in range(max_iters):
        for filter in filters:
            filtered_signal = lfilter(filter,1.0,signal)
            period = estimate_period(filtered_signal)
            modes.append(filtered_signal)

        if len(modes)>= n:
            break

    return modes[:n]

#调用函数与绘图
n = 5
modes = fmd(signal_df,n)

#检查模态数据
for i,mode in enumerate(modes):
    print(f'Mode {i+1}: Max={np.max(mode)},Min={np.min(mode)}')

#绘制结果
plt.figure(figsize=(10,8))
plt.subplot(len(modes)+ 1, 1, 1)
plt.plot(time,signal_df['Signal'].values)
#plt.plot(time,signal_df.values)
plt.title('Original Signal')

for i,mode in enumerate(modes,start=1):
    plt.subplot(len(modes)+1, 1, i+1)
    plt.plot(time,mode)
    plt.title(f'Mode {i}')

plt.tight_layout()
plt.show()
