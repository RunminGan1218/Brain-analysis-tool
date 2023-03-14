import matplotlib.pyplot as plt
import numpy as np
from .preprocessing import butter_filt
from .WaveAnalysisFun import *
from scipy import signal

def plot_vector_field(vec, norm=True, fig=None,ax=None, show_flag=True):
    #######
    # INPUT
    # ph - complex-valued scalar field representing vector directions
    # norm - whether to represent the norm of the vec 
    # OUTPUT
    # vector field plot
    #######

    x = np.arange(vec.shape[1])+0.5
    y = np.arange(vec.shape[0])+0.5
    if norm:
        u = np.real(vec)
        v = np.imag(vec)
    else:
        u = np.real(np.exp(1j*np.angle(vec)))
        v = np.imag(np.exp(1j*np.angle(vec)))

    if ax == None:
        if fig==None:
            fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.quiver(x, y, u, -v, color = 'k',scale=2,scale_units='xy',width=0.005)
    ax.invert_yaxis()
    if show_flag:
        plt.show()
    return fig,ax


# # 绘制每个通道图像
def show_channelwave(sig, grishape, fig=None, show_flag=True):
    #######
    # input:
    # sig.shape = (channels,time) 
    #######
    if fig==None:
        fig = plt.figure()
    for id in range(sig.shape[0]):
        ax = fig.add_subplot(grishape[0],grishape[1],id+1)
        ax.plot(sig[id])
    if show_flag:
        plt.show()
    return fig


# wave of each channel 
def channel_wave(trial_sig, fs, time_points, lowband=[3,6], highband=[30,50],period=None, fig=None, fig1=None, show_flag=True):
    #######
    # input:
    # trial_sig:shape=(trial,time) type=np.array
    #######
    if fig == None:
        fig = plt.figure()
    if fig1 == None:
        fig1 = plt.figure()
    if period == None:  
        period=[0,trial_sig.shape[1]]
    ax0 = fig.add_subplot(111)
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212)

    low_t_sig = butter_filt(trial_sig,fs,lowband,'bandpass')
    high_t_sig = butter_filt(trial_sig,fs,highband,'bandpass')

    # 每个试次图像
    for trial in trial_sig.shape[0]:
        ax0.plot(time_points,trial_sig[trial],color='0.1')
        ax1.plot(time_points[int(period[0]):int(period[1])],low_t_sig[trial][int(period[0]):int(period[1])],color='0.1')
        ax2.plot(time_points[int(period[0]):int(period[1])],high_t_sig[trial][int(period[0]):int(period[1])],color='0.1')
        # ax3.plot(time_points,channel_trial_wave2_[id][trial],color='0.1')
    # 试次间均值
    ax0.plot(time_points,np.mean(trial_sig,0),color='red')
    ax1.plot(time_points[int(period[0]):int(period[1])],np.mean(low_t_sig[:,int(period[0]):int(period[1])],0),color='red')
    ax2.plot(time_points[int(period[0]):int(period[1])],np.mean(high_t_sig[:,int(period[0]):int(period[1])],0),color='red')
    # ax3.plot(time_points,np.mean(channel_trial_wave2_[id],0),color='red')
    # 标准差
    ax0.plot(time_points,np.mean(trial_sig,0)+np.std(trial_sig,0),color='red',linestyle='--')
    ax0.plot(time_points,np.mean(trial_sig,0)-np.std(trial_sig,0),color='red',linestyle='--')
    ax0.set_title('channel trial-mean LFP')
    fig1.suptitle('channel trial-mean LFP at different frequences')
    ax1.set_title('low-frequency')
    ax2.set_title('high-frequency')
    if show_flag:
        plt.show()
    return fig,fig1

# # ITPC
# # ITPC循环单通道（一次显示一个channellist中的通道）
def ITPC_channel(trial_sig, fs, time_points, period, band=[1,70], fig=None, show_flag=True):
    #######
    # input:
    # trial_sig:shape=(trial,time) type=np.array
    #######
    frequencies = np.logspace(np.log10(band[0]),np.log10(band[1]))
    itpc = np.zeros((len(frequencies),period))
    for idx,f in enumerate(frequencies):
        itpc[idx] = ITPC(trial_sig,fs,f)
    
    if fig == None:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    levels = [0,0.2,0.4,0.6,0.8,1]
    ax_ = ax.pcolormesh(time_points,frequencies,itpc,cmap='jet',vmin=0)
    # plt.contourf(time_points,frequencies,itpc_mean,100,cmap='jet',vmax=1,vmin=0, extend='min')
    ax.set_yscale('symlog')
    fig.colorbar(ax_,ax=ax,ticks=levels,label='ITPC')
    # plt.yticks([1,3,5,10,30,50])
    ax.set_title('ITPC channel')
    if show_flag:
        plt.show()
    return fig

# # ITPC_location
def ITPC_location(ct_sig,t_interval,fs,freq_band,channel_list,gridshape,fig=None,show_flag=True):
    ###########
    # INPUT:
    # ct_sig:(channels,trials,time)
    interval = [1.5,4]

    itpc_location = []
    for id in channel_list:
        frequencies = np.linspace(freq_band[0],freq_band[1],40)
        itpc = np.zeros(len(frequencies),)

        for idx,f in enumerate(frequencies):
            itpc[idx] = np.mean(ITPC(ct_sig[id,:,int(interval[0]*fs):int(interval[1]*fs)],fs,f))
        
        itpc_location.append(np.mean(itpc))
    itpc_location = np.array(itpc_location).reshape(gridshape[0],gridshape[1])
    if fig==None:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    ax_ = ax.matshow(itpc_location,cmap='jet')
    # plt.gca().invert_yaxis()
    ax.set_title('ITPC of each location')
    fig.colorbar(ax_,ax=ax,label='ITPC')
    if show_flag:
        plt.show()
    return fig


# # Average filtered LFP traveling wave-like behavior
def plot_traveling_wave_behavior(ct_sig, fs, t_period, travel_axis_list, fig=None,show_flag=True):
    # INPUT:
    # ct_sig:shape(channels,trials,time)

    if fig == None:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    t_pnt = np.array(t_period)*fs
    tw_mat = np.zeros((len(travel_axis_list),ct_sig.shape[2]))
    for i,id in enumerate(travel_axis_list):
        tw_mat[i] = np.mean(ct_sig[id],0)

    ax_ = ax.pcolor(tw_mat[:,int(t_pnt[0]):int(t_pnt[1])], cmap='PRGn')
    ax.set_title('Normalized low-frequency Wave')
    ax.invert_yaxis()
    fig.colorbar(ax_,ax = ax, label='Voltage (μV)')
    if show_flag:
        plt.show()
    return fig


# Phase-amplitude coupling MI
def Phase_amplitude_coupling(trial_sig_low, trial_sig_high, n_bin = 20, fig=None, show_flag=True):
    # INPUT:
    # trial_sig:shape(trials,time)
    # low: baseband
    # high: carrier
    if fig==None:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    analytic_sig_low = signal.hilbert(trial_sig_low)
    analytic_sig_high = signal.hilbert(trial_sig_high)

    mid_interval = np.arange(-np.pi,np.pi,2*np.pi/n_bin)+np.pi/n_bin

    Phase = np.angle(analytic_sig_low)
    Amp = np.abs(analytic_sig_high)
    MI, MeanAmp = PAC_MI(Phase,Amp,n_bin)

    ax.bar(mid_interval,MeanAmp)
    ax.set_title(f'channel  MI = {MI:.4f}')
    ax.set_ylabel('Amp')
    ax.set_xlabel('Phase')

    if show_flag:
        plt.show()
    return fig

    
def MI_maxAmpPhase_location(ct_sig_low, ct_sig_high, channel_list, bad_channel_list, gridshape, n_bin=20, fig=None, fig1=None, show_flag=True):
    # Phase Coupling MI

    if fig==None:
        fig = plt.figure()
    if fig1==None:
        fig1 = plt.figure()
    
    analytic_low = signal.hilbert(ct_sig_low)
    analytic_high = signal.hilbert(ct_sig_high)

    MI_Gr = np.zeros((len(channel_list),))
    MeanAmp_ct = np.zeros((len(channel_list),n_bin))
    for loc,id in enumerate(channel_list):
        Phase = np.angle(analytic_low[id])
        Amp = np.abs(analytic_high[id])
        MI_Gr[loc], MeanAmp_ct[id] = PAC_MI(Phase,Amp,n_bin)

        if id in bad_channel_list:
            MI_Gr[loc] = float('inf')
    MI_Gr = MI_Gr.reshape(gridshape[0],gridshape[1])


    ax = fig.add_subplot(111)
    ax_ = ax.pcolor(MI_Gr ,cmap='rainbow')
    ax.set_title('Phase Coupling MI')
    ax.invert_yaxis()
    fig.colorbar(ax_,ax = ax,label='MI')
    
    mid_interval = np.arange(-np.pi,np.pi,2*np.pi/n_bin)+np.pi/n_bin
    PA_Gr = np.zeros(len(channel_list))
    for loc,id in enumerate(channel_list):
        bin_num = int(np.array(np.where(MeanAmp_ct[id] == np.max(MeanAmp_ct[id]))))
        PA_Gr[loc] = mid_interval[bin_num]
        if id in bad_channel_list:
            PA_Gr[loc] = float('inf')
    PA_Gr = PA_Gr.reshape((gridshape[0],gridshape[1]))

    PA_ax = fig1.add_subplot(111)
    PA_ax_ = PA_ax.pcolor(PA_Gr, vmin=-np.pi, vmax=np.pi, cmap='gist_rainbow')
    PA_ax.set_title('low_fre Phase with Maximum high_fre Amplitude')
    PA_ax.invert_yaxis()
    fig1.colorbar(PA_ax_,ax = PA_ax,ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],label='Phase of low-frequency oscillation')
    
    if show_flag:
        plt.show()
    return fig,fig1

# # Spacial wave length and speed
def waveLen_waveSpeed(ct_sig_low, ct_sig_high, fs, stim_start, channel_list, pixel_spacing_x, pixel_spacing_y, Thresh, gridshape, n_mode, tt, fig=None, show_flag=True):  
    
    # INPUT:
    # tt: 取信号比较明显的时刻作为计算时间相位的时间点
    
    SVD_tc_data = ct_sig_low.swapaxes(0,1)
    SVD_tc_data1 = ct_sig_high.swapaxes(0,1)
    
    period = SVD_tc_data.shape[2]

    most_resp_mode = np.zeros((SVD_tc_data.shape[0],),dtype=int)

    most_resp_mode1 = np.zeros((SVD_tc_data.shape[0],),dtype=int)

    
    low_waveLength_trials = np.zeros((SVD_tc_data.shape[0],gridshape[0],gridshape[1],SVD_tc_data.shape[2]))
    low_waveVelocity_trials = np.zeros((SVD_tc_data.shape[0],gridshape[0],gridshape[1],SVD_tc_data.shape[2]))
    high_waveLength_trials = np.zeros((SVD_tc_data.shape[0],gridshape[0],gridshape[1],SVD_tc_data.shape[2]))
    high_waveVelocity_trials = np.zeros((SVD_tc_data.shape[0],gridshape[0],gridshape[1],SVD_tc_data.shape[2]))

    for trial in range(SVD_tc_data.shape[0]):
        SVDout, SpatialAmp, SpatialPhase, TemporalAmp, TemporalPhase, = complexSVD(SVD_tc_data[trial],n_mode)
        _, _, most_resp_mode[trial] = TemporalImportance_Amp(TemporalAmp,stim_start,period,Thresh=Thresh,baselineEnd=0)
        # print(most_resp_mode[trial])
        rec_a = SVDModeReconstruct(SVDout,most_resp_mode[trial])
        ft, signIF= instantaneous_frequency(rec_a,fs)
        rec_a_rct = reorgChannels(rec_a,channel_list,gridshape)
        # print(rec_a_rct.shape)
        pm,pd,dx,dy = phase_gradient_complex_multiplication(rec_a_rct,pixel_spacing_x,pixel_spacing_y,signIF)
        # print(pd.shape)


        # wave length and velocity
        # Spacial wavelength: reciprocal of spatial frequency  空间频率倒数
        # spatial frequency：梯度模值/2*pi
        # velocity：Spacial wavelength*temporal frequency

        ft_rct = reorgChannels(ft,channel_list,gridshape)
        low_waveLength_trials[trial] = 1/pm
        low_waveVelocity_trials[trial] = ft_rct/pm
        
        # 重复一遍高频信号
        SVDout, SpatialAmp, SpatialPhase, TemporalAmp, TemporalPhase, = complexSVD(SVD_tc_data1[trial],n_mode)
        _, _, most_resp_mode1[trial] = TemporalImportance_Amp(TemporalAmp,stim_start,period,Thresh=Thresh,baselineEnd=0)
        # print(most_resp_mode[trial])
        # mostRespSPhase1[trial] = SpatialPhase[:,most_resp_mode1[trial]]
        rec_a = SVDModeReconstruct(SVDout,most_resp_mode1[trial])
        ft, signIF= instantaneous_frequency(rec_a,fs)
        rec_a_rct = reorgChannels(rec_a,channel_list,gridshape)
        # print(rec_a_rct.shape)
        pm,pd,dx,dy = phase_gradient_complex_multiplication(rec_a_rct,pixel_spacing_x,pixel_spacing_y,signIF)
        # print(pd.shape)
        ft_rct = reorgChannels(ft,channel_list,gridshape)
        
        high_waveLength_trials[trial] = 1/pm
        high_waveVelocity_trials[trial] = ft_rct/pm


    # 绘制波长波速直方图
    fig = plt.figure(1)
    f1_ax1 = fig.add_subplot(121)
    low_wl = low_waveLength_trials[:,:,:,tt].reshape(-1)
    low_wl = low_wl[low_wl<10]
    high_wl = high_waveLength_trials[:,:,:,tt].reshape(-1)
    high_wl = high_wl[high_wl<10]
    num_bins = 100
    n, bins, patches = f1_ax1.hist(low_wl,num_bins, density=True,facecolor='blue', alpha=0.5) 
    n, bins, patches = f1_ax1.hist(high_wl,num_bins, density=True,facecolor='red', alpha=0.5) 
    # f1_ax1.set_xscale('log')
    f1_ax1.legend(['low-freq wave','high-freq wave'])
    f1_ax1.set_xlabel('WaveLength(mm/cycle)') 
    f1_ax1.set_ylabel('Probability') 
    f1_ax1.set_title('Histogram of WaveLength') 

    f1_ax2 = fig.add_subplot(122)
    low_wv = low_waveVelocity_trials[:,:,:,tt].reshape(-1)
    low_wv = low_wv[low_wv<200]
    low_wv = low_wv[low_wv>0]/1000
    high_wv = high_waveVelocity_trials[:,:,:,tt].reshape(-1)
    high_wv = high_wv[high_wv<200]
    high_wv = high_wv[high_wv>0]/1000
    # print(low_waveLength_trials[1,:,:,tt])
    num_bins = 100
    n, bins, patches = f1_ax2.hist(low_wv,num_bins, density=True,facecolor='blue', alpha=0.5) 
    n, bins, patches = f1_ax2.hist(high_wv,num_bins, density=True,facecolor='red', alpha=0.5) 
    # f1_ax2.set_xscale('log')
    f1_ax2.legend(['low-freq wave','high-freq wave'])
    f1_ax2.set_xlabel('WaveVelocity(m/s)') 
    f1_ax2.set_ylabel('Probability') 
    f1_ax2.set_title('Histogram of WaveVelocity') 

    if show_flag:
        plt.show()
    return fig


    # Spacial PhaseDiff and gradient
def spacial_phase_gradient(ct_sig, fs, stim_start, fast_response_channel, channel_list, bad_channel_list, pixel_spacing_x, pixel_spacing_y, Thresh, gridshape, n_mode, tt, fig=None, show_flag=True):
    SVD_tc_data = ct_sig.swapaxes(0,1)
    period = SVD_tc_data.shape[2]

    Tamp = np.zeros((SVD_tc_data.shape[0],SVD_tc_data.shape[2],n_mode))
    sig = np.zeros((SVD_tc_data.shape[0],n_mode))
    most_resp_mode = np.zeros((SVD_tc_data.shape[0],),dtype=int)
    mostRespSPhase = np.zeros((SVD_tc_data.shape[0],SVD_tc_data.shape[1]))
    pd_trials = np.zeros((SVD_tc_data.shape[0],gridshape[0],gridshape[1],SVD_tc_data.shape[2]))
    

    for trial in range(SVD_tc_data.shape[0]):
        SVDout, SpatialAmp, SpatialPhase, TemporalAmp, TemporalPhase, = complexSVD(SVD_tc_data[trial],n_mode)
        Tamp[trial], sig[trial], most_resp_mode[trial] = TemporalImportance_Amp(TemporalAmp,stim_start,period,Thresh=Thresh,baselineEnd=0)
        # print(most_resp_mode[trial])
        mostRespSPhase[trial] = SpatialPhase[:,most_resp_mode[trial]]
        rec_a = SVDModeReconstruct(SVDout,most_resp_mode[trial])
        ft, signIF= instantaneous_frequency(rec_a,fs)
        rec_a_rct = reorgChannels(rec_a,channel_list,gridshape)
        # print(rec_a_rct.shape)
        pm,pd,dx,dy = phase_gradient_complex_multiplication(rec_a_rct,pixel_spacing_x,pixel_spacing_y,signIF)
        # print(pd.shape)
        pd_trials[trial] = pd

        # wave length and velocity
        # Spacial wavelength: reciprocal of spatial frequency  空间频率倒数
        # spatial frequency：梯度模值/2*pi
        # velocity：Spacial wavelength*temporal frequency

   
    # spacial phase gradient
    gradientDirection_array_tt = np.angle(np.mean(np.exp(1j*pd_trials[:,:,:,tt]),0))
    gradientDirection_var_array_tt = 1 - np.abs(np.mean(np.exp(1j*pd_trials[:,:,:,tt]),0))
    vec = gradientDirection_var_array_tt*np.exp(1j*gradientDirection_array_tt)
    # print(vec)
    

    AlignedAngles, AlignedVar, Pvals = mostRespSVDphaseDiff(mostRespSPhase,fast_response_channel)
    AlignedAngles_Gr = np.zeros(AlignedAngles.shape)
    AlignedVar_Gr = np.zeros(AlignedVar.shape)
    Pvals_Gr = np.zeros(Pvals.shape)

    for loc,id in enumerate(channel_list):
        AlignedAngles_Gr[loc] = AlignedAngles[id]
        AlignedVar_Gr[loc] = AlignedVar[id]
        Pvals_Gr[loc] = Pvals[id]
        if (id in bad_channel_list or Pvals[id]<0.0006):
            AlignedAngles_Gr[loc] = -float('inf')
            vec[loc//gridshape[1],loc%gridshape[1]]=0
    AlignedAngles_Gr = AlignedAngles_Gr.reshape(gridshape)
    AlignedVar_Gr = AlignedVar_Gr.reshape(gridshape)
    Pvals_Gr = Pvals_Gr.reshape(gridshape)
    # print(Pvals_Gr)
    # levels = np.arange(-np.pi, np.pi, 0.001)
    if fig==None:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    ax_ = ax.pcolor(AlignedAngles_Gr,vmin=-np.pi, vmax=np.pi ,cmap='rainbow')
    ax.set_title('Phase Gridient')
    # ax.set_title('high-frequency Phase Gridient')
    fig.colorbar(ax_,ax = ax,ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],label='Phase')
    _, ax = plot_vector_field(vec,norm=True,ax=ax,show_flag=False)

    if show_flag:
        plt.show()
    return fig, Pvals_Gr
    # # p值显示
    # plt.matshow(Pvals_Gr)
    # plt.title('Pvals')
    # plt.show()

def spacial_phase_diff(ct_sig, stim_start, channel_diff,Thresh,n_mode,fig=None,show_flag=True):
    # # spacial phase difference
    if fig==None:
        fig=plt.figure()
    
    SVD_tc_data = ct_sig.swapaxes(0,1)
    period = SVD_tc_data.shape[2]
    mostRespSPhase = np.zeros((SVD_tc_data.shape[0],SVD_tc_data.shape[1]))
    

    for trial in range(SVD_tc_data.shape[0]):
        SVDout, SpatialAmp, SpatialPhase, TemporalAmp, TemporalPhase, = complexSVD(SVD_tc_data[trial],n_mode)
        _, _, most_resp_mode = TemporalImportance_Amp(TemporalAmp,stim_start,period,Thresh=Thresh,baselineEnd=0)
        # print(most_resp_mode[trial])
        mostRespSPhase[trial] = SpatialPhase[:,most_resp_mode]


    bin_num = 20
    ph_dif = (mostRespSPhase[:,channel_diff[0]]-mostRespSPhase[:,channel_diff[1]])
    ph_dif[ph_dif<0] = ph_dif[ph_dif<0]+2*np.pi
    
    phaseDiff_pl_ax = fig.add_subplot(111,projection='polar')
    bars = phaseDiff_pl_ax.hist(ph_dif,bin_num,(0,2*np.pi))
    phaseDiff_pl_ax.set_title('Phase difference(low-frequency)')
    # phaseDiff_pl_ax.set_title('Phase difference(high-frequency)')
    if show_flag:
        plt.show()
    return fig


