import numpy as np
import scipy.signal as signal
import scipy.integrate as integrate


#Low pass Butterworth filter function
def Butterworth_filter_LP(sampling_Time, order, cut_off_frequency, Input_Signal):


    fs = 1/(sampling_Time)
    b, a = signal.butter(order, cut_off_frequency/(fs/2))
    return signal.filtfilt(b, a, Input_Signal, axis=0)


#High pass Butterworth filter function
def Butterworth_filter_HP(sampling_Time, order, cut_off_frequency, Input_Signal):


    fs = 1/(sampling_Time)
    b, a = signal.butter(order, cut_off_frequency/(fs/2), btype='highpass')
    return signal.filtfilt(b, a, Input_Signal, axis=0)

#Numerical integration
def Numerical_integration(Xdata, Ydata):

    dxmean = np.mean(np.diff(Xdata, n=1, axis=0))

    return integrate.cumtrapz(Ydata, None, dxmean, axis=0)


