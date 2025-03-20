import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, hilbert, correlate
from scipy.signal.windows import hann
from scipy.io import loadmat
from numpy.linalg import inv


def myfft(fs, x, plot_mode=False):
    """FFT计算与MATLAB版本等效"""
    N = len(x)
    ff = np.linspace(0, fs, N + 1)[:N]
    amp = np.abs(np.fft.fft(x)/N*2)
    amp = amp[:N // 2]
    ff = ff[:N // 2]
    if plot_mode:
        plt.plot(ff, amp, 'b')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    return ff, amp


def TT(y, fs):
    """周期估计函数"""
    M = fs
    na = correlate(y, y, mode='full', method='auto')[-M - 1:]
    na = na[len(na) // 2:]

    # 寻找第一个过零点
    zeroposi = None
    for lag in range(1, len(na)):
        if na[lag - 1] > 0 and na[lag] < 0:
            zeroposi = lag
            break
    if zeroposi is None:
        zeroposi = 0

    na = na[zeroposi:]
    max_position = np.argmax(na)
    T = zeroposi + max_position
    return T


def CK(x, T, M=2):
    """相关峭度计算"""
    x = x.flatten()
    N = len(x)
    x_shift = np.zeros((M + 1, N))
    x_shift[0] = x.copy()

    for m in range(1, M + 1):
        x_shift[m, T:] = x_shift[m - 1, :-T]

    product = np.prod(x_shift, axis=0)
    ck = np.sum(product ** 2) / (np.sum(x ** 2) ** (M + 1))
    return ck


def max_IJ(X):
    """找到矩阵中最大元素的位置"""
    tempI = np.argmax(X, axis=0)
    M = np.max(X, axis=0)
    J = np.argmax(M)
    I = tempI[J]
    return I, J, X[I, J]


def xxc_mckd(fs, x, f_init, term_iter, T=None, M=3, plot_mode=False):
    """MCKD核心算法"""
    x = x.flatten()
    L = len(f_init)
    N = len(x)

    # 如果 T 为 None，通过信号的包络估计周期
    if T is None:
        xx_envelope = np.abs(hilbert(x)) - np.mean(np.abs(hilbert(x)))
        T = TT(xx_envelope, fs)

    T = int(round(T))  # 确保 T 是整数

    # 初始化 XmT
    XmT = np.zeros((L, N, M + 1))
    for m in range(M + 1):
        for l in range(L):
            if l == 0:
                XmT[l, m * T:, m] = x[:N - m * T]
            else:
                XmT[l, 1:, m] = XmT[l - 1, :-1, m]

    Xinv = inv(XmT[:, :, 0] @ XmT[:, :, 0].T)

    f = f_init.copy()
    ck_best = 0
    y_final = np.zeros((N, term_iter))
    f_final = np.zeros((L, term_iter))
    ck_iter = []
    T_final = T

    for n in range(term_iter):
        y = (f.T @ XmT[:, :, 0]).T

        # 生成 yt 矩阵
        yt = np.zeros((N, M + 1))
        yt[:, 0] = y.flatten()
        for m in range(1, M + 1):
            yt[m * T:, m] = yt[:N - m * T, m - 1]

        # 计算 alpha 和 beta
        alpha = np.zeros((N, M + 1))
        for m in range(M + 1):
            cols = [k for k in range(M + 1) if k != m]
            alpha[:, m] = np.prod(yt[:, cols], axis=1) ** 2 * yt[:, m]

        beta = np.prod(yt, axis=1)

        # 计算 Xalpha
        Xalpha = np.zeros(L)
        for m in range(M + 1):
            Xalpha += XmT[:, :, m] @ alpha[:, m]  # 修正：逐维度计算

        # 更新滤波器系数
        f = (np.sum(y ** 2) / (2 * np.sum(beta ** 2))) * Xinv @ Xalpha
        f /= np.sqrt(np.sum(f ** 2))

        # 计算 CK 值
        ck = np.sum(np.prod(yt, axis=1) ** 2) / (np.sum(y ** 2) ** (M + 1))
        ck_iter.append(ck)

        # 更新周期估计
        xy_envelope = np.abs(hilbert(y)) - np.mean(np.abs(hilbert(y)))
        T = TT(xy_envelope, fs)
        T = int(round(T))  # 确保 T 是整数
        T_final = T

        # 更新 XmT 矩阵
        XmT = np.zeros((L, N, M + 1))
        for m in range(M + 1):
            for l in range(L):
                if l == 0:
                    XmT[l, m * T:, m] = x[:N - m * T]
                else:
                    XmT[l, 1:, m] = XmT[l - 1, :-1, m]

        Xinv = inv(XmT[:, :, 0] @ XmT[:, :, 0].T)
        y_final[:, n] = np.convolve(x, f, mode='same')
        f_final[:, n] = f

    return y_final, f_final, np.array(ck_iter), T_final


def FMD(fs, x, filter_size, cut_num, mode_num, max_iter_num):
    """特征模式分解主函数"""
    freq_bound = np.arange(0, 1, 1 / cut_num)

    # 初始化滤波器组
    temp_filters = np.zeros((filter_size, cut_num))
    for n in range(cut_num):
        # 使用 firwin 生成滤波器系数，numtaps 设置为 filter_size
        temp_filters[:, n] = firwin(
            filter_size,  # 修正：numtaps 设置为 filter_size
            [freq_bound[n] + 1e-15, freq_bound[n] + 1 / cut_num - 1e-15],
            window='hann',  # 使用字符串 'hann' 表示 Hanning 窗口
            pass_zero=False
        )

    # 其余代码保持不变
    result = {'iterations': []}
    temp_sig = np.tile(x, (cut_num, 1)).T

    iter_count = 0
    while True:
        iter_num = max_iter_num - (cut_num - mode_num) * 2 if iter_count == 0 else 2

        current_iter = {
            'modes': [],
            'filters': [],
            'corr_matrix': None,
            'location': None,
            'output': None
        }

        for n in range(cut_num):
            y_iter, f_iter, _, T_iter = xxc_mckd(
                fs, temp_sig[:, n], temp_filters[:, n], iter_num, None, 1, False)

            current_iter['modes'].append({
                'y': y_iter[:, -1],
                'f': f_iter[:, -1],
                'T': T_iter
            })

        # 更新信号和滤波器
        temp_sig = np.array([mode['y'] for mode in current_iter['modes']]).T
        temp_filters = np.array([mode['f'] for mode in current_iter['modes']]).T

        # 计算相关系数矩阵
        corr_matrix = np.abs(np.corrcoef(temp_sig.T))
        corr_matrix = np.triu(corr_matrix, 1)
        I, J, _ = max_IJ(corr_matrix)

        # 模式合并逻辑
        ki = CK(temp_sig[:, I], current_iter['modes'][I]['T'])
        kj = CK(temp_sig[:, J], current_iter['modes'][J]['T'])
        output = J if ki > kj else I

        # 删除被合并的模式
        temp_sig = np.delete(temp_sig, output, axis=1)
        temp_filters = np.delete(temp_filters, output, axis=1)
        cut_num -= 1

        if cut_num == mode_num - 1:
            break

        iter_count += 1

    final_modes = temp_sig
    return final_modes


# 使用示例
if __name__ == "__main__":
    # 加载数据（假设x.mat文件存在）
    data = loadmat('x.mat')
    x = data['x'].flatten()
    fs = 20000
    t = np.arange(len(x)) / fs

    # 原始信号绘图
    plt.figure()
    plt.plot(t, x)
    plt.title('Time waveform of mixed signal')

    plt.figure()
    myfft(fs, x, plot_mode=True)
    plt.title('FFT spectrum of mixed signal')

    # FMD分解
    filtersize = 30
    cutnum = 7
    modenum = 2
    maxiternum = 20

    final_modes = FMD(fs, x, filtersize, cutnum, modenum, maxiternum)

    # 结果可视化
    plt.figure()
    for i in range(final_modes.shape[1]):
        plt.subplot(final_modes.shape[1], 1, i + 1)
        plt.plot(t, final_modes[:, i])
        plt.title(f'Mode {i + 1}')

    plt.figure()
    for i in range(final_modes.shape[1]):
        plt.subplot(final_modes.shape[1], 1, i + 1)
        ff, amp = myfft(fs, final_modes[:, i])
        plt.plot(ff, amp / amp.max())
        plt.xlim(0, 1000)

    plt.figure()
    for i in range(final_modes.shape[1]):
        plt.subplot(final_modes.shape[1], 1, i + 1)
        envelope = np.abs(hilbert(final_modes[:, i])) - np.mean(np.abs(hilbert(final_modes[:, i])))
        ff, amp = myfft(fs, envelope)
        plt.plot(ff, amp)
        plt.xlim(0, 300)

    plt.show()