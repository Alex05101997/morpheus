# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:55:19 2019

@author: Alex
"""
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.fftpack import fft, fftfreq, ifft
from scipy.signal import find_peaks, windows, resample_poly, firwin, freqz
import numpy as np
from cmath import exp
from math import gcd

#next power of two after n(used for zero-padding)
def power_of_two(n):
    i = 0
    while 2**i<=n:
        i += 1
    return 2**i

def quadratic_interpolation(y1,y2,y3):
    p = 1/2*(y1-y3)/(y1-2*y2+y3)
    yp = y2 - 1/4*(y1-y3)*p
    return(p,yp)
    
def linear_interpolation(y1, y2, n):
    y = (1-n)*y1 + n*y2
    return y
def spectral_mixing(s1,s2,n):
    mix = []
    mix_arr = np.asarray(mix,dtype=complex)
    s1_arr = np.asarray(s1,dtype=complex)
    s2_arr = np.asarray(s2,dtype=complex)
    for elem1,elem2 in zip(s1_arr,s2_arr):
        mix_arr = np.append(mix_arr,(1-n)*elem1+n*elem2)
    return mix_arr
#read from the disk
fs_ga, wav_ga = read('GA001.wav')
wav_pad_ga = np.pad(wav_ga,(0,power_of_two(wav_ga.size)-wav_ga.size),'constant')
k=wav_pad_ga.size/fs_ga
hamming_1 = windows.hamming(int(4192*k))
hamming_1_pad = np.pad(hamming_1,(0,wav_pad_ga.size-int(4192*k)),'constant')
#print(hamming_1.size)
wav_pad_ga=wav_pad_ga * hamming_1_pad
#print(wav_ga.size)
#fft
y_ga = fft(wav_pad_ga)
freq_ga = fftfreq(y_ga.size,1/fs_ga)
#the magnitude spectrum and the peaks of the magnitude spectrum
magnitude_ga = abs(y_ga)
peaks_ga, properties_ga = find_peaks(magnitude_ga[int(70*k):int(8000*k)], distance=int(80*k))
peaks_ga_cor = [x+int(70*k) for x in peaks_ga]
#the phase spectrum
phase_ga = np.unwrap(np.angle(y_ga))
#fundamental frequency and harmonics and magnitudes of the ff and of the harmonics
frequencies_ga=[]
frequencies_ga_mag=[]
for peak_ga in peaks_ga_cor:
    p_ga,yp_ga = quadratic_interpolation(magnitude_ga[peak_ga-int(k)], magnitude_ga[peak_ga],magnitude_ga[peak_ga+int(k)])
    frequencies_ga.append((peak_ga+p_ga*k)*fs_ga/magnitude_ga.size)
    frequencies_ga_mag.append(yp_ga)
ff_ga = frequencies_ga[0]
#print(ff_ga)
#print(harmonics_ga)
#print(harmonics_magnitude_ga)
#linear interpolation for the ff phase
ff_ga_phase = linear_interpolation(phase_ga[int(ff_ga*k)], phase_ga[int(ff_ga*k)+int(k)],ff_ga-int(ff_ga))
#phases of the harmomics(linear interpolation) 
harmonics_phase_ga = []
for frequency_ga in frequencies_ga[1:]:
    interpolated_phase_ga = linear_interpolation(phase_ga[int(frequency_ga*k)], phase_ga[int(frequency_ga*k)+1],frequency_ga-int(frequency_ga))
    harmonics_phase_ga.append(interpolated_phase_ga)
#print(ff_ga_phase)
#print(harmonics_phase_ga)

fs_cc, wav_cc=read('CC103.wav')
#dividing the CC recording into 25 ms long frames
hamming_2 = windows.hamming(400)
frames = []
for i in range(wav_cc.size//400):
    frames.append(wav_cc[i*400:(i+1)*400])
frames_new = []
frames_new_arr = np.asarray(frames_new)
mag = []
mag_arr = np.asarray(mag)

for frame in frames:
    frame = frame * hamming_2
    frame_pad = np.pad(frame,(0,power_of_two(frame.size)-frame.size),'constant')
    k_1 = frame_pad.size/fs_cc
    y_cc = fft(frame_pad)
    freq_cc = fftfreq(y_cc.size,1/fs_cc)
    magnitude_cc = abs(y_cc)
    peaks_cc, properties_cc = find_peaks(magnitude_cc[int(70*k_1):int(8000*k_1)], distance=int(80*k_1))
    peaks_cc_cor = [x+int(70*k_1) for x in peaks_cc]
    bins = peaks_cc_cor
    phase_cc = np.unwrap(np.angle(y_cc))
    
    frequencies_cc=[]
    frequencies_cc_mag=[]
    for peak_cc in peaks_cc_cor:
        p_cc,yp_cc = quadratic_interpolation(magnitude_cc[peak_cc-1], magnitude_cc[peak_cc],magnitude_cc[peak_cc+1])
        frequencies_cc.append((peak_cc+p_cc*k_1)*fs_cc/magnitude_cc.size)
        frequencies_cc_mag.append(yp_cc)
    ff_cc = frequencies_cc[0]
    
    #print(frequencies_cc)
    #print(ff_cc)
    
    
    #linear interpolation for the ff phase
    ff_cc_phase = linear_interpolation(phase_cc[int(ff_cc*k_1)], phase_cc[int(ff_cc*k_1)+1],ff_cc-int(ff_cc))
    #phases of the harmomics(linear interpolation) 
    harmonics_phase_cc = []
    for frequency_cc in frequencies_cc:
        interpolated_phase_cc = linear_interpolation(phase_cc[int(frequency_cc*k_1)], phase_cc[int(frequency_cc*k_1)+1],frequency_cc-int(frequency_cc))
        harmonics_phase_cc.append(interpolated_phase_cc)
    #print(ff_cc_phase)
    #print(harmonics_phase_cc)
    
    
    #mapping indices
    mapping_ind = []
    frequency_ratio = int(round(ff_cc))/int(round(ff_ga))
    for frequency_cc in frequencies_cc:
        mapping_ind.append(int((frequencies_cc.index(frequency_cc))/frequency_ratio+0.5))
    
    #frequency shifts
    freq_shifts = []
    for map_ind in mapping_ind:
        d_i = int(ff_cc*(map_ind-mapping_ind.index(map_ind))*(frame_pad.size/fs_cc)+0.5)
        freq_shifts.append(d_i)
    #print(freq_shifts)
    
    #gains
    gains = []
    for mag_ga, mag_cc in zip(frequencies_ga_mag, frequencies_cc_mag):
        gains.append(mag_cc/mag_ga)
    
    #phase corrections
    phase_cor = []
    for ph_ga, ph_cc in zip(harmonics_phase_ga, harmonics_phase_cc):
        phase_cor.append(ph_cc/ph_ga)
        
    #resampled spectrum
    upsample_factor = int(round(ff_cc))
    downsample_factor = int(round(ff_ga))
    new_fs = frequency_ratio*fs_cc
    up_fs = upsample_factor*fs_cc
    #gcd_fs = fs_cc*upsample_factor//gcd(upsample_factor, downsample_factor)
    ff_max = max(int(round(ff_cc)),int(round(ff_ga)))
    cutoff_frequency = up_fs/(2*ff_max)
    cutoff_frequency_n = cutoff_frequency*2/(up_fs)
    fir_filter = firwin(240,cutoff_frequency_n,window='hamming')    
    wav_ga_resampled = resample_poly(wav_ga,int(round(ff_cc)),int(round(ff_ga)),window=fir_filter)
    #hamming_3 = windows.hamming(wav_ga_resampled.size)
    #wav_ga_resampled = wav_ga_resampled*hamming_3
    y_res = fft(wav_ga_resampled)
    freq_res = fftfreq(y_res.size,1/new_fs)
    new_bins = [int((freq * y_res.size)/new_fs) for freq in frequencies_cc]
    #plt.figure(2)
    #w,h = freqz(fir_filter, fs=up_fs)
    #plt.plot(w[:10],abs(h[:10]))
    #print(fir_filter)
    #harmonics mapping and filtering
    #print(y_res.size,y_cc.size)
    synthesis_spectrum = np.zeros(y_cc.size,dtype=complex)
    #for i in range(len(bins)-1) :
        #if synthesis_spectrum[bins[i]:bins[i+1]].shape==y_res[int(frequencies_cc[1:][i]):int(frequencies_cc[1:][i])].shape:
            #synthesis_spectrum[bins[i]:bins[i+1]] = y_res[bins[i]+freq_shifts[i]:bins[i+1]+freq_shifts[i+1]]#*gains[i]*exp(1j*phase_cor[i])
    for b, f, d, g, ph in zip(bins, new_bins, freq_shifts, gains, phase_cor):
            synthesis_spectrum[b] = y_res[int(f)+d]*g*exp(1j*ph)
            #print(b)
            #print(int(f)+d)
    for value in synthesis_spectrum[0:synthesis_spectrum.size//2]:
        index = list(synthesis_spectrum[0:synthesis_spectrum.size//2]).index(value)
        synthesis_spectrum[synthesis_spectrum.size-index-1] = value
    #print(frequencies_cc)     
    #spectral mixing
    mix_arr = np.empty(y_cc.size,dtype=complex)
    mix_arr = spectral_mixing(y_cc,synthesis_spectrum,1)
    #print(y_res[new_bins])
    #mag_arr = np.concatenate((mag_arr,mix_arr))
    #ifft
    new_frame = ifft(mix_arr)
    #print(ff_cc)
    frames_new_arr = np.concatenate((frames_new_arr,new_frame))
#print(ff_ga)
write('GM1.wav',fs_cc,frames_new_arr.astype('int16'))
'''
plt.figure()
plt.plot(freq_ga[0:freq_ga.size//2],magnitude_ga[0:magnitude_ga.size//2])
plt.figure()
plt.plot(freq_ga[0:freq_ga.size//2],phase_ga[0:phase_ga.size//2])
'''
plt.figure()
plt.plot((np.arange(wav_cc.size//2)/wav_cc.size)*fs_cc,abs(fft(wav_cc))[0:wav_cc.size//2])
#plt.figure()
#plt.plot((np.arange(wav_cc.size//2)/wav_cc.size)*fs_cc,np.unwrap(np.angle(fft(wav_cc)))[0:wav_cc.size//2])

plt.figure()
plt.plot((np.arange(frames_new_arr.size)/frames_new_arr.size)*fs_cc,abs(fft(frames_new_arr))[0:frames_new_arr.size])

#plt.figure()
#plt.plot((np.arange(frames_new_arr.size//2)/frames_new_arr.size)*fs_cc,np.unwrap(np.angle(fft(frames_new_arr)))[0:frames_new_arr.size//2])

plt.figure()
plt.plot(freq_ga,magnitude_ga)
#plt.plot(freq_ga[peaks_ga_cor][0:3],magnitude_ga[peaks_ga_cor][0:3],'x')

plt.figure()
plt.plot(frames_new_arr)
plt.figure()
plt.plot(wav_ga)
plt.figure()
plt.plot(wav_cc)
plt.show()
#print(mag_arr.size,fft(wav_cc).size)
#print(np.array_equal(mag_arr[0:10000],fft(wav_cc[0:10000])))
#print(frequencies_ga)

