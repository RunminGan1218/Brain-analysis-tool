import numpy as np
import matplotlib.pyplot as plt
import pywt


# 需要分析的四个频段
# iter_freqs = [
#     {'name': 'Delta', 'fmin': 0, 'fmax': 4},
#     {'name': 'Theta', 'fmin': 4, 'fmax': 8},
#     {'name': 'Alpha', 'fmin': 8, 'fmax': 13},
#     {'name': 'Beta', 'fmin': 13, 'fmax': 35},
# ]

iter_freqs = [
    {'name': 'low-freq', 'fmin': 4, 'fmax': 7},
    {'name': 'high-freq', 'fmin': 30, 'fmax': 60},
]

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


######################################################连续小波变换##########
# totalscal小波的尺度，对应频谱分析结果也就是分析几个（totalscal-1）频谱
def TimeFrequencyCWT(data,fs,totalscal,wavelet='cgau8'):
    # 采样数据的时间维度
    t = np.arange(data.shape[0])/fs
    # 中心频率
    wcf = pywt.central_frequency(wavelet=wavelet)
    # 计算对应频率的小波尺度
    cparam = 2 * wcf * totalscal
    scales = cparam/np.arange(totalscal, 1, -1)
    # print(scales)
    # 连续小波变换
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavelet, 1.0/fs)
    # print(frequencies)
    # 绘图
    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(t, data)
    plt.xlabel(u"time(s)")
    plt.title(u"Time spectrum")
    plt.subplot(212)
    plt.contourf(t, frequencies, abs(cwtmatr))
    plt.ylabel(u"freq(Hz)")
    plt.xlabel(u"time(s)")
    plt.subplots_adjust(hspace=0.4)
    plt.show()

########################################小波包变换-重构造分析不同频段的特征(注意maxlevel，如果太小可能会导致)#########################
def TimeFrequencyWP(data, fs, wavelet, maxlevel = 8):
    # 小波包变换这里的采样频率为250，如果maxlevel太小部分波段分析不到
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    # 频谱由低到高的对应关系，这里需要注意小波变换的频带排列默认并不是顺序排列，所以这里需要使用’freq‘排序。
    freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
    # print(len(freqTree))
    # 计算maxlevel最小频段的带宽
    freqBand = fs/(2*2**maxlevel)
    #######################根据实际情况计算频谱对应关系，这里要注意系数的顺序
    # 绘图显示
    fig, axes = plt.subplots(len(iter_freqs)+1, 1, figsize=(10, 7), sharex=True, sharey=False)
    # 绘制原始数据
    axes[0].plot(data)
    axes[0].set_title('原始数据')
    for iter in range(len(iter_freqs)):
        # 构造空的小波包
        new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
        for i in range(len(freqTree)):
            # 第i个频段的最小频率
            bandMin = i * freqBand
            # 第i个频段的最大频率
            bandMax = bandMin + freqBand
            # 判断第i个频段是否在要分析的范围内
            if (iter_freqs[iter]['fmin']<=bandMin and iter_freqs[iter]['fmax']>= bandMax):
                # 给新构造的小波包参数赋值
                new_wp[freqTree[i]] = wp[freqTree[i]].data
                # print(bandMin,bandMax)
            # else:
            #     new_wp[freqTree[i]] = np.zeros_like(wp[freqTree[i]].data)
        # 绘制对应频率的数据
        axes[iter+1].plot(new_wp.reconstruct(update=True))
        # 设置图名
        axes[iter+1].set_title(iter_freqs[iter]['name'])
    plt.show()


#############################################小波包计算四个频段的能量分布
def WPEnergy(data, fs, wavelet, maxlevel=6):
    # 小波包分解
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    # 频谱由低到高的对应关系，这里需要注意小波变换的频带排列默认并不是顺序排列，所以这里需要使用’freq‘排序。
    freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
    # 计算maxlevel最小频段的带宽
    freqBand = fs / (2*2 ** maxlevel)
    # 定义能量数组
    energy = []
    # 循环遍历计算四个频段对应的能量
    for iter in range(len(iter_freqs)):
        iterEnergy = 0.0
        for i in range(len(freqTree)):
            # 第i个频段的最小频率
            bandMin = i * freqBand
            # 第i个频段的最大频率
            bandMax = bandMin + freqBand
            # 判断第i个频段是否在要分析的范围内
            if (iter_freqs[iter]['fmin'] <= bandMin and iter_freqs[iter]['fmax'] >= bandMax):
                # 计算对应频段的累加和
                iterEnergy += pow(np.linalg.norm(wp[freqTree[i]].data, ord=None), 2)
        # 保存四个频段对应的能量和
        energy.append(iterEnergy)
    # 绘制能量分布图
    plt.plot([xLabel['name'] for xLabel in iter_freqs], energy, lw=0, marker='o')
    plt.title('能量分布')
    plt.show()


# if __name__ == '__main__':
#     # 读取筛选好的epoch数据
#     epochsCom = mne.read_epochs(r'F:\BaiduNetdiskDownload\BCICompetition\BCICIV_2a_gdf\Train\Fif\A02T_epo.fif')

#     dataCom = epochsCom[10].get_data()[0][0][0:1024]
#     TimeFrequencyCWT(dataCom, fs=250, totalscal=10, wavelet='cgau8')
#     TimeFrequencyWP(dataCom, 250, wavelet='db4', maxlevel=8)
#     WPEnergy(dataCom, fs=250, wavelet='db4', maxlevel=6)






