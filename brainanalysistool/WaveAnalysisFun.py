import numpy as np
from scipy import signal
from pactools import Comodulogram, REFERENCES
import pycwt as wavelet
import matplotlib.pyplot as plt



class SVD:
    def __init__(self,U,S,V):
        self.U = U
        self.S = S
        self.V = V

def reorgChannels(signal,channel_list,channel_shape=(8,8)):
    # reorgnize the channels of signal(channels*time) from channel_list
    reorg = np.zeros(signal.shape,dtype=signal.dtype)
    for loc,id in enumerate(channel_list):
        reorg[loc] = signal[id]
    reorg = reorg.reshape(channel_shape[0],channel_shape[1],-1)
    return reorg

def complexSVD(data,N_modes):
    # do SVD on analytic signal
    # data must be 2-D array
    # N_modes must smaller than min(size(data))
    analytic_signal = signal.hilbert(data)
    U,S,VH=np.linalg.svd(analytic_signal,full_matrices=False) 
    V = VH.T.conjugate()

    A = U*S
    A = A[:,:N_modes]
    SpatialAmp=np.abs(A)  
    SpatialPhase=np.angle(A)
    B = V*S
    B = B[:,:N_modes]
    TemporalAmp=np.abs(B)  
    TemporalPhase=np.angle(B)
    SVDout = SVD(U,S,V)

    return SVDout, SpatialAmp, SpatialPhase, TemporalAmp, TemporalPhase

def SVDModeReconstruct(SVD,mode):
    # input
    # SVD:  SVD(class SVD) of the original analytic signal(channels*time) 
    # mode:  mode to extract
    # output
    # rec_a:  reconstruct analytic signal(channels*time)
    u = SVD.U[:,mode].reshape(-1,1)
    s = SVD.S[mode]
    vh = SVD.V[:,mode].conjugate().reshape(1,-1)
    rec_a = u.dot(s*vh)
    return rec_a


def TemporalImportance_Amp(Tamp, Stim_time, resp_end,  Thresh, baselineEnd=1):

    if baselineEnd:
        stopBase = Tamp.shape[0]
    else:
        stopBase = Stim_time

    baseM = np.mean(Tamp[:stopBase],0)
    baseS = np.std(Tamp[:stopBase],0)

    ZTamp = (Tamp-baseM)/baseS
    Stim = ZTamp[Stim_time:resp_end]

    sig = np.zeros(Tamp.shape[1])
    for mod in range(Tamp.shape[1]):
        if np.any(Stim[:,mod]>Thresh):
            sig[mod] = 1
    
    most_resp_mode = int(np.array(np.where(np.max(Stim,0)==np.max(Stim))))
    
    return ZTamp, sig, most_resp_mode

    



def circ_r(alpha):
    r = np.sum(np.exp(1j*alpha))
    r = np.abs(r)/len(alpha)
    return r

def circ_rtest(alpha):
    r = circ_r(alpha)
    n = len(alpha)
    R = n*r
    z = R**2 / n
    pval = np.exp(np.sqrt(1+4*n+4*(n**2-R**2))-(1+2*n))
    return pval, z
    

def SVDphaseDiff(SpatialPhase,ref_id):
    for i in range(SpatialPhase.shape[0]):
        SpatialPhase[i] = SpatialPhase[i]-SpatialPhase[i,ref_id]
    AlignedAngles = np.angle(np.mean(np.exp(1j*SpatialPhase),0))
    AlignedVar = 1-np.abs(np.mean(np.exp(1j*SpatialPhase),0))

    Pvals = np.zeros((SpatialPhase.shape[1],SpatialPhase.shape[2]))
    for chan in range(SpatialPhase.shape[1]):
        for mod in range(SpatialPhase.shape[2]):
            p, _ = circ_rtest(SpatialPhase[:,chan,mod].reshape(-1))
            Pvals[chan,mod] = p

    return AlignedAngles, AlignedVar, Pvals

def mostRespSVDphaseDiff(mostRespSPhase,ref_id):
    # mostRespSPhase shape:Trial*Spacial_channels
    # output:
    # AlignedAngles  shape:(Spacial_channels)
    # AlignedVar     shape:(Spacial_channels)
    # Pvals          shape:(Spacial_channels)


    mostRespSPhase = mostRespSPhase-mostRespSPhase[:,ref_id].reshape(-1,1)
    AlignedAngles = np.angle(np.mean(np.exp(1j*mostRespSPhase),0))
    AlignedVar = 1-np.abs(np.mean(np.exp(1j*mostRespSPhase),0))

    Pvals = np.zeros(mostRespSPhase.shape[1])
    for chan in range(mostRespSPhase.shape[1]):
        p, _ = circ_rtest(mostRespSPhase[:,chan])
        Pvals[chan] = p

    return AlignedAngles, AlignedVar, Pvals


def instantaneous_frequency(analytic_x, Fs):
    # analytic_x shape(channels,t)
    thresh_tol = 0.9

    ft = np.zeros(analytic_x.shape)
    ft[:,0:-1] = np.angle( analytic_x[:,1:]*analytic_x[:,0:-1].conjugate() )  / (2*np.pi/Fs)

    if  np.sum(ft>0)/np.sum(ft!=0) > thresh_tol:		 # positive frequency
        signIF = 1 
    elif np.sum(ft<0)/np.sum(ft!=0) > thresh_tol:		 # negative frequency
        signIF = -1 
    else: 																	 # mixed/indeterminate
        signIF = float('inf')

    return ft, signIF

def phase_gradient_complex_multiplication(analytic_signal, pixel_spacing_x=0.4, pixel_spacing_y=0.65,signIF=1):
    # analytic_signal must be reshape to datacube(r,c,t) r*c=channels
    # init
    shape = analytic_signal.shape
    pm = np.zeros(shape) 
    pd = np.zeros(shape) 
    dx = np.zeros(shape) 
    dy = np.zeros(shape) 


    dx[:,0] = np.angle(analytic_signal[:,1]*analytic_signal[:,0].conjugate())/pixel_spacing_x
    dx[:,-1] = np.angle(analytic_signal[:,-1]*analytic_signal[:,-2].conjugate())/pixel_spacing_x
    dx[:,1:-1] = np.angle(analytic_signal[:,2:]*analytic_signal[:,:-2].conjugate())/(2*pixel_spacing_x)
    dx = -dx*signIF

    dy[0,:] = np.angle(analytic_signal[1,:]*analytic_signal[0,:].conjugate())/pixel_spacing_y
    dy[-1,:] = np.angle(analytic_signal[-1,:]*analytic_signal[-2,:].conjugate())/pixel_spacing_y
    dy[1:-1,:] = np.angle(analytic_signal[2:,:]*analytic_signal[:-2,:].conjugate())/(2*pixel_spacing_y)
    dy = -dy*signIF

    pm = np.sqrt( dx**2 + dy**2 ) / (2*np.pi)
    pd = np.angle( 1j*dy+dx )
    return pm,pd,dx,dy

def PACComodulogram(ax,signal,fs,low_fq_range,low_fq_width=2.0,high_fq_range='auto', high_fq_width='auto',method='tort'):
    estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
                             low_fq_width=low_fq_width, high_fq_range=high_fq_range, high_fq_width=high_fq_width, method=method,
                             progress_bar=False)
    estimator.fit(signal)
    estimator.plot(titles=[REFERENCES[method]], axs=[ax])
    return estimator

def PAC_MI(Phase, Amp, n_bin):
    # input
    # Phase, Amp:shape(trials,times)  time interval: post_stim 1000s recommended

    win_size = 2*np.pi/n_bin
    n_trial = Phase.shape[0]
    position = np.arange(-np.pi,np.pi,win_size)
    
    MeanAmp = np.zeros((n_trial,n_bin))
    for i, l_bound in enumerate(position):
        for trial in range(n_trial):
            c1 = Phase[trial]>l_bound
            c2 = Phase[trial]<l_bound+win_size
            MeanAmp[trial,i] = np.mean(Amp[trial][c1&c2])
    MeanAmp = np.mean(MeanAmp,0)
    q = MeanAmp/np.sum(MeanAmp)
    MI = (np.log(n_bin)-(-np.sum(q*np.log(q))))/np.log(n_bin)
    return MI,MeanAmp


def wavelet_filt(data, fs, band=[], mode='bandpass'):
    dt = 1/fs
    freqs = np.logspace(np.log10(1),np.log10(fs/2))
    # print(freqs)
    s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
    dj = 1 / 12  # Twelve sub-octaves per octaves
    J = 7 / dj  # Seven powers of two with dj sub-octaves
    mother = wavelet.Morlet(6)
    wave, scales, f, coi, fft, fftfreqs = wavelet.cwt(data, dt, dj, s0, J,
                                                      mother)
    # print(wave.shape)
    # print(scales.shape)
    print(f.shape)
    print(f)

    # if mode=='bandpass':
    iwave = wavelet.icwt(wave, scales, dt, dj, mother)
    # print(iwave)
    plt.figure()
    plt.plot(data, label = 'data')
    plt.plot(iwave,color = 'k', label = 'data_rec')
    plt.legend()
    plt.show()

def ITPC(data, fs, centerfreq):
    def conv_len_pow2(n):
        i = 0
        while(pow(2,i)<n):
            i+=1
        return pow(2,i)
    n_wavelet     = data.shape[1]
    n_data        = data.shape[1]*data.shape[0]
    n_convolution = n_wavelet+n_data-1
    n_conv_pow2   = conv_len_pow2(n_convolution)

    time    = np.linspace(-n_wavelet/fs/2,n_wavelet/fs/2-1/fs,n_wavelet)
    wavelet = np.exp(2*1j*np.pi*centerfreq*time) * np.exp(-np.square(time)/(2*(np.square(4/(2*np.pi*centerfreq)))))/centerfreq



    datafft = np.fft.fft(data.reshape(-1),n_conv_pow2)

    dataconv = np.fft.ifft(np.fft.fft(wavelet,n_conv_pow2)*datafft)
    dataconv = dataconv[0:n_convolution]
    dataconv = dataconv[int((n_wavelet-1)/2)-1:-2-int((n_wavelet-1)/2)].reshape(-1,n_wavelet)

    itpc = np.abs(np.mean(np.exp(1j*np.angle(dataconv)),0))
    return itpc







if __name__ == '__main__':
    data = np.random.rand(64,1000,5000)
    # print(data)
    data = data.swapaxes(0,1)
    # print(data)
    
    START = 0
    END = 5000
    PERIOD = END - START

    Thresh = 4.5


    n_mode = 10
    Tamp = np.zeros((data.shape[0],data.shape[2],n_mode))
    sig = np.zeros((data.shape[0],n_mode))
    most_resp_mode = np.zeros((data.shape[0],),dtype=int)
    mostRespSPhase = np.zeros((data.shape[0],data.shape[1]))

    for trial in range(data.shape[0]):
        SVDout, SpatialAmp, SpatialPhase, TemporalAmp, TemporalPhase, = complexSVD(data[trial],n_mode)
        Tamp[trial], sig[trial], most_resp_mode[trial] = TemporalImportance_Amp(TemporalAmp,-START+1000,PERIOD,Thresh=Thresh,baselineEnd=1)
        # print(most_resp_mode[trial])
        mostRespSPhase[trial] = SpatialPhase[:,most_resp_mode[trial]]

        # print(Tamp[trial].shape)
        # print(sig[trial])
        # print(most_resp_mode[trial])
        
    AlignedAngles, AlignedVar, Pvals = mostRespSVDphaseDiff(mostRespSPhase,11-1)
    print(AlignedAngles)
    print(AlignedVar)
    print(Pvals)
    # print(data.T.conjugate())
    # print(signal.hilbert(data))
    # SVDout, SpatialAmp, SpatialPhase, TemporalAmp, TemporalPhase, = complexSVD(data,5)
    # U,S,V= SVDout
    # print(U[0].dot(S[0].reshape(-1,1)*VH[0]))
    # print(U)
    # print(S)
    # print(V)
    # print(SpatialAmp)
    # print(SpatialPhase)
    # print(TemporalAmp)
    # print(TemporalPhase)


    