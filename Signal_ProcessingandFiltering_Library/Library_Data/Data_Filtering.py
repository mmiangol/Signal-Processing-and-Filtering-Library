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

# Piecewise interpolation
def Piecewise_Interoplation(xdata, ydata, interpolation_steps):

    # Parameter initialization
    xmin = np.min(xdata)
    xmax = np.max(xdata)
    step_size = ((xmax - xmin) / interpolation_steps)
    xinter = np.linspace(xmin, xmax, interpolation_steps+1)
    xstart = xmin
    xend = xmin + step_size
    yinter = np.zeros(interpolation_steps+1)
    ytot = np.zeros(interpolation_steps*2)


    # Extraction of each individual piecewise section
    for count in range(interpolation_steps):
        ind = np.where(np.logical_and(xdata >= xstart, xdata <= xend))

        fitted_data = np.polyfit(list(xdata[ind]), list(ydata[ind]), 1)
        y1 = np.polyval(fitted_data, xstart)
        y2 = np.polyval(fitted_data, xend)

        ytot[count*2] = y1
        ytot[count * 2+1] = y2

        xstart = xstart + step_size
        xend = xend + step_size

    # Piecewise section merge
    for count in range(interpolation_steps-1):

        yinter[count+1] = ((ytot[count*2+1])+(ytot[count*2+2]))/2

    yinter[0] = ytot[0]
    yinter[interpolation_steps] = ytot[-1]
    return xinter, yinter

# Numerical derivation
def Numerical_derivation(xdata, ydata):

    dx = np.gradient(xdata)
    dy = np.gradient(ydata)

    return np.divide(dy, dx)

# Fast Fourier Transformation plot data
def FFT_plot(timestep, ydata):

    length = ydata.size
    freq = np.fft.fftfreq(length, d=timestep)[:length//2]
    amplitude = np.fft.fft(ydata)
    yamplitude = 2.0/length*np.abs(amplitude[0:length//2])
    return freq, yamplitude

