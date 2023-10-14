import numpy as np
import scipy.signal as signal
import scipy.integrate as integrate


#Low pass Butterworth filter function
def Butterworth_filter_LP(sampling_time, order, cut_off_frequency, input_signal):

    fs = 1/(sampling_time)
    b, a = signal.butter(order, cut_off_frequency/(fs/2))
    return signal.filtfilt(b, a, input_signal, axis=0)


#High pass Butterworth filter function
def Butterworth_filter_HP(sampling_time, order, cut_off_frequency, input_signal):

    fs = 1/(sampling_time)
    b, a = signal.butter(order, cut_off_frequency/(fs/2), btype='highpass')
    return signal.filtfilt(b, a, input_signal, axis=0)


#Numerical integration
def Numerical_integration(xdata, ydata):
    
    dxmean = np.mean(np.diff(xdata, n=1, axis=0))
    Int_Y = integrate.cumtrapz(ydata, None, dxmean, axis=0)
    Int_Y = np.concatenate((np.array([0]), Int_Y), axis=0)
    return Int_Y


#Polynomial detrend
def Polynomial_detrend(xdata, ydata, order):

    Trend = np.polyfit(list(xdata), list(ydata), order)
    yp = np.polyval(Trend, xdata)
    return np.subtract(ydata, yp)



