# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:55:19 2019

@author: Alex
"""
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.fftpack import fft, fftfreq, ifft
from scipy.signal import find_peaks, windows, resample_poly, firwin, freqz
from scipy.interpolate import interp1d
import numpy as np
from cmath import exp

#next power of two after n(used for zero-padding)
def power_of_two(n):
    i = 0
    while 2**i<=n:
        i += 1
    return 2**i

def quadratic_interpolation(f,m):
    p = 1/2*(f[m-1]-f[m+1])/(f[m-1]-2*f[m]+f[m+1])
    yp = f[m] - 1/4*(f[m-1]-f[m+1])*p
    return(p,yp)

#read from the disk
fs_ga, wav_ga = read('GA1.wav')
#zero-padding
wav_pad_ga = np.pad(wav_ga,power_of_two(wav_ga.size)-wav_ga.size,'constant')
#fft
y_ga = fft(wav_pad_ga)
freq_ga = fftfreq(y_ga.size,1/fs_ga)
#the magnitude spectrum and the peaks of the magnitude spectrum
magnitude_ga = abs(y_ga)
peaks_ga, properties_ga = find_peaks(magnitude_ga)
#the phase spectrum
phase_ga = np.unwrap(np.angle(y_ga))
#fundamental frequency and harmonics and magnitudes of the ff and of the harmonics
p_ga,yp_ga = quadratic_interpolation(magnitude_ga[peaks_ga], np.argmax(magnitude_ga[peaks_ga]))
ff_ga = abs((np.argmax(magnitude_ga[peaks_ga])+p_ga)*fs_ga/magnitude_ga[peaks_ga].size)
j = 2
harmonics_ga = []
harmonics_magnitude_ga = []
while ff_ga*j < fs_ga/2:
    harmonics_ga.append(ff_ga*j)
    harmonics_magnitude_ga.append(magnitude_ga[peaks_ga][int(round(ff_ga*j*magnitude_ga[peaks_ga].size/fs_ga))])
    j += 1
#linear interpolation
phase_inter_ga = interp1d(freq_ga[0:freq_ga.size//2],phase_ga[0:phase_ga.size//2],fill_value='extrapolate')
#phases of the ff and of the harmonics
ff_phase_ga = phase_inter_ga(ff_ga)
#print(ff_phase)
harmonics_phase_ga = phase_inter_ga(harmonics_ga)
#print(harmonics_phase)

fs_cc, wav_cc=read('CC1.wav')
#dividing the CC recording into 25 ms long frames
hamming = windows.hamming(400)
frames = []
for i in range(wav_cc.size//400):
    frames.append(wav_cc[i*400:(i+1)*400]*hamming)
frames_new = []
frames_new_arr = np.asarray(frames_new)
for frame in frames:
    wav_pad_cc = np.pad(frame,power_of_two(wav_cc.size)-wav_cc.size,'constant')
    y_cc = fft(wav_pad_cc)
    freq_cc = fftfreq(y_cc.size,1/fs_cc)
    magnitude_cc = abs(y_cc)
    peaks_cc, properties_cc = find_peaks(magnitude_cc)
    phase_cc = np.unwrap(np.angle(y_cc))
    p_cc,yp_cc = quadratic_interpolation(magnitude_cc[peaks_cc], np.argmax(magnitude_cc[peaks_cc]))
    ff_cc = abs((np.argmax(magnitude_cc[peaks_cc])+p_cc)*fs_cc/magnitude_cc[peaks_cc].size)
    
    #print(ff_cc)
    l = 2
    harmonics_cc = []
    harmonics_magnitude_cc = []
    harmonic_bins = []
    while ff_cc*l < fs_cc/2:
        harmonics_cc.append(ff_cc*l)
        harmonics_magnitude_cc.append(magnitude_cc[peaks_cc][int(round(ff_cc*l*magnitude_cc[peaks_cc].size/fs_cc))])
        harmonic_bins.append(int(round(ff_cc*l*magnitude_cc[peaks_cc].size/fs_cc)))
        l += 1
    #print(harmonics_cc)
    #print(harmonics_bins)
    phase_inter_cc = interp1d(freq_cc[0:freq_cc.size//2],phase_cc[0:phase_cc.size//2],fill_value='extrapolate')
    ff_phase_cc = phase_inter_cc(ff_cc)
    harmonics_phase_cc = phase_inter_cc(harmonics_cc)
    #w,h = freqz(fir_filter, fs=1648000)
    #plt.plot(w,abs(h))
    
    #mapping indices
    mapping_ind = []
    frequency_ratio = int(round(ff_cc))/int(round(ff_ga))
    for harmonic in harmonics_cc:
        mapping_ind.append(int(harmonics_cc.index(harmonic)/frequency_ratio+0.5))
    #print(mapping_ind)
    
    #frequency shifts
    freq_shifts = []
    for map_ind in mapping_ind:
        d_i = int(ff_cc*(map_ind-mapping_ind.index(map_ind))*frame.size/fs_cc+0.5)
        freq_shifts.append(d_i)
    #print(freq_shifts)
    
    #gains
    gains = []
    for mag_ga, mag_cc in zip(harmonics_magnitude_ga, harmonics_magnitude_cc):
        gains.append(mag_cc/mag_ga)
    
    #phase corrections
    phase_cor = []
    for ph_ga, ph_cc in zip(harmonics_phase_ga, harmonics_phase_cc):
        phase_cor.append(ph_cc/ph_ga)
    
    #resampled spectrum
    upsample_factor = int(round(ff_cc))
    new_fs = frequency_ratio*fs_cc
    cutoff_frequency = new_fs/(fs_cc*upsample_factor)
    fir_filter = firwin(80,cutoff_frequency,window='hamming')    
    wav_ga_resampled = resample_poly(wav_ga,int(round(ff_cc)),int(round(ff_ga)),window=fir_filter)
    y_res = fft(wav_ga_resampled)
    freq_res = fftfreq(y_res.size,1/new_fs)
    
    #harmonics mapping and filtering
    synthesis_spectrum = []
    for b, d, g, ph in zip(harmonic_bins, freq_shifts, gains, phase_cor):
        synthesis_spectrum.append(y_res[b+d]*g*exp(j*ph))
        
    #spectral mixing
    mix = interp1d(harmonics_cc[0:len(synthesis_spectrum)],synthesis_spectrum,fill_value='extrapolate')
    mix_spectrum = mix(freq_cc)   
    #ifft
    new_frame = ifft(mix_spectrum)
    frames_new_arr = np.append(frames_new_arr,new_frame)
'''write('GM1.wav',fs_cc,frames_new_arr.astype(wav_ga.dtype))
plt.plot(frames_new_arr)

 
#plotting the magnitude spectrum and the peaks of the magnitude spectrum(ga recording)
plt.figure(1)
plt.plot(freq_ga[0:freq_ga.size//2],magnitude_ga[0:magnitude_ga.size//2])
plt.plot(freq_ga[peaks_ga][0:freq_ga[peaks_ga].size//2],magnitude_ga[peaks_ga][0:magnitude_ga[peaks_ga].size//2],'x')
#plotting the phase spectrum(ga recording)
plt.figure(2)
plt.plot(freq_ga[0:freq_ga.size//2],phase_ga[0:phase_ga.size//2])
#plotting the magnitude spectrum and the peaks of the magnitude spectrum(cc recording)
plt.figure(3)
plt.plot(freq_cc[0:freq_cc.size//2],magnitude_cc[0:magnitude_cc.size//2])
plt.plot(freq_cc[peaks_cc][0:freq_cc[peaks_cc].size//2],magnitude_cc[peaks_cc][0:magnitude_cc[peaks_cc].size//2],'x')
#plotting the phase spectrum(cc recording)
plt.figure(4)
plt.plot(freq_cc[0:freq_cc.size//2],phase_cc[0:phase_cc.size//2])
plt.plot(harmonics_cc,phase_inter_cc(harmonics_cc))
#plotting the magnitude spectrum of the resampled signal
plt.figure(5)
plt.plot(freq_res[0:freq_res.size//2],abs(y_res[0:y_res.size//2]))
#mixed spectrum
plt.figure(6)   
plt.plot(harmonics_cc,abs(np.asarray(synthesis_spectrum)))     
plt.plot(harmonics_cc,abs(mix_spectrum))
plt.show()
'''






















