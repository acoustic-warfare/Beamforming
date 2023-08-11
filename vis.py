# VISUALIZE MICROPHONE ARRAY SIGNALS
#

import numpy as np
import matplotlib.pyplot as plt
import math
import config_test as config
import active_microphones as am
import calc_r_prime as crp

def delete_mic_data(signal, mic_to_delete):
    #   sets signals from selected microphones in mic_to_delete to 0
    new_signal = signal.copy()
    new_signal[:,mic_to_delete] = 0
    return new_signal

def plot_all_mics(data, N_mics = 4*64, amp_lim=0, samp_lim = 0):
    #   plot of data from all microphones
    plt.figure()
    for i in range(N_mics): 
        plt.plot(data[:,i], c=cmap(i/N_mics))
    if amp_lim:
        #plt.suptitle('All microphones', fontsize = FS_title)
        plt.ylim([-max_value*1.1, max_value*1.1]); filename = 'all_sigs' + '_del_mics'
    else:
        #plt.title('All microphones, no amplitude limit', fontsize = FS_title)
        filename = 'all_sigs_nolim'
    if samp_lim: plt.xlim([0, plot_samples])
    plt.xlabel('Samples', fontsize = FS_label), plt.ylabel('Amplitude', fontsize = FS_label)
    plt.xticks(fontsize= FS_tics), plt.yticks(fontsize= FS_tics)
    plt.tight_layout()
    if save_fig:
        plt.savefig('plots/signal_analyzis/'+filename+'.png', dpi = 500, format = 'png')

def plot_all_individual(data, start_val, a):
    #   plot of all individual signals in subplots, two periods
    fig, axs = plt.subplots(rows, cols, figsize=(5,7))
    fig.tight_layout(pad = 0.1)
    plt.subplots_adjust(left=0.03, bottom=0, right=0.97,
                        top=0.9, wspace=0.1, hspace=0.7)
    fig.suptitle("Individual signals, A"+str(a+1), fontsize=FS_title)
    for j in range(rows):
        for i in range(cols):
            axs[7-j,i].plot(data[start_val:start_val+plot_samples, \
                                 i+j*cols+array_elements*a], \
                                 c=cmap((i+j*cols)/(array_elements)))
            axs[7-j,i].set_title(str(i+j*cols+array_elements*a), fontsize=FS_mics)
            axs[7-j,i].set_ylim(-max_value*1.1, max_value*1.1)
            axs[7-j,i].axis('off')
    if save_fig:
        filename = 'indi_sig_A'+str(a+1)
        plt.savefig('plots/signal_analyzis/'+filename+'.png', dpi = 500, format = 'png')

def plot_selected(data, plot_mics, amp_lim=0, samp_lim = 0, subtitle=''):
    # --- PLOT ---
    #   of selected microphones, given by plot_mics
    arr_plot_mics = np.array(plot_mics)     # convert plot_mics to numpy array with correct index
    plt.figure()
    for i in range(len(arr_plot_mics)):
        plt.plot(data[:,int(arr_plot_mics[i])], label=f"{arr_plot_mics[i]}", c=cmap(i/(len(plot_mics)-1+0.1)))
    if amp_lim: # amplitude limitation
        #plt.suptitle('Selected microphones', fontsize = FS_title)
        plt.ylim([-max_value*1.1, max_value*1.1])
    else: # no amplitude limitation
        #plt.suptitle('Selected microphones, no amplitude limit', fontsize = FS_title)
        filename = 'sel_sig_nolim_M'+str(plot_mics)
    if samp_lim: # sample limitation
        plt.xlim([0, plot_samples])
    
    if subtitle != '': plt.title(subtitle, fontsize = FS_title*0.75) # extra subtitle
     
    if len(plot_mics) < 12:
        plt.legend(loc = 4, fontsize = FS_label)
    plt.xlabel('Samples',fontsize = FS_label),  plt.ylabel('Amplitude',fontsize = FS_label)
    plt.xticks(fontsize=FS_tics),               plt.yticks(fontsize=FS_tics)
    plt.tight_layout()
    if save_fig:
        filename = 'sel_sig_M'+str(plot_mics)
        plt.savefig('plots/signal_analyzis/'+filename+'.png', dpi = 500, format = 'png')

def plot_energy(data, mics_FFT):
    #   plot of FFT of several signals
    samples = len(data[:,0])
    t_stop = samples/fs
    t = np.linspace(0,t_stop,samples)
    arr_mics_FFT = np.array(mics_FFT,dtype=int)
    plt.figure()
    for i in range(len(arr_mics_FFT)):
        data_FFT = np.fft.fft(data[:,int(arr_mics_FFT[i])])
        energy = abs(data_FFT)**2
        freq = np.fft.fftfreq(t.shape[-1])
        plt.plot(fs*freq, energy, label=f"{arr_mics_FFT[i]}", c=cmap(i/(len(mics_FFT))))
    
    plt.suptitle('Energy of selected microphones signals', fontsize = FS_title)
    plt.xlabel('Frequency [Hz]', fontsize = FS_label)
    plt.xticks(fontsize= FS_tics), plt.yticks(fontsize= FS_tics)
    plt.legend()
    plt.tight_layout()

def plot_array(ignore_mic = [], mode=1):
    #   plot array setup
    #   displays mode, and if microphones are ignored or not
    active_mics = am.active_microphones(mode, 'all')  # load active microphones
    r_prime_all, r_prime = crp.calc_r_prime(config.ELEMENT_DISTANCE)

    fig, ax = plt.subplots(figsize=(config.COLUMNS*config.ACTIVE_ARRAYS/2, config.ROWS/2))
    ax.set_box_aspect(int(config.ROWS)/int(config.COLUMNS*config.ACTIVE_ARRAYS))            # aspect ratio
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)              # remove top and right axis lines
    plt.tight_layout()

    color_arr = ['r', 'b', 'g', 'm']   # element colors for the separate arrays
    dx = 0.001                              # offset of element index text
    dy = 0.001                              # offset of element index text

    element = 0
    for array in range(config.ACTIVE_ARRAYS):
        plt.title('Array setup')
        for mic in range(config.ROWS*config.COLUMNS):
            x = r_prime_all[0,element]
            y = r_prime_all[1,element]
            if element in zero_mics: #(np.sum(data[element], axis = 0)):
                ax.scatter(x, y, color = 'none', edgecolor='k', linewidths = 1)
            elif element in ignore_mic:
                ax.scatter(x, y, color = '#fdff38', edgecolor='#fdff38', linewidths = 1)
            elif element in active_mics:
                ax.scatter(x, y, color = color_arr[array])
            else:
                ax.scatter(x, y, color = 'none', edgecolor=color_arr[array], linewidths = 1)
            plt.text(x-dx, y+dy, str(element))
            element += 1
    if save_fig:
        filename = 'array_setup_signal_analyzing'
        plt.savefig('plots/signal_analyzis/'+filename+'.png', dpi = 500, format = 'png')


# name of file of recordings, should be .npy file
filename = '4.npy' 

# Plot options
show_plots = 1              # if show_plots = 1, then plots will be shown, if = 0 no plots will be shown
                            #   show (=1) or hide (=0) different type of plots
all_mics = 1                # plots of all microphones
selected = 1                # plots of several selected microphones
all_individual = 1          # plot of individual microphones
energy = 0                  # plot of energy of signals
plot_array_setup = 1        # plot the array setup
save_fig = 1                # if 1, save figure

color_map_type = 'jet'      # 'plasma', 'cool', 'inferno'
plot_period = 3             # periods to plot
f0 = 1000                   # frequency of recorded sinus signal
fs = 48828                  # sampling frequency

plot_samples = math.floor(plot_period*(fs/f0))  # number of samples to plot, to use for axis scaling

# Plot text sizes
FS_title = 18
FS_mics = 15
FS_label = 15
FS_tics = 15

rows = config.ROWS                      # number of rows in one array
cols = config.COLUMNS                   # number of columns in one array
array_elements = rows*cols              # total elements in one array

cmap = plt.get_cmap(color_map_type)     # type of color map to use for coloring the different signals

# load data, from file
directory = 'recordings'
data = np.load(directory+'/'+ filename)
data = data.T
data = data[2000:,:]                # take out selected samples

# find the mics that sends zeros
data_sum = np.sum(data,axis=0)
zero_mics = np.where(data_sum[np.arange(0,64*3)]==0)[0]

# Different max values of the microphone amplitudes
max_value = np.max(np.abs(data))                    # maximum value of all microphones
max_each_mic = np.max(data[:,:rows*cols*config.ACTIVE_ARRAYS], axis=0)
max_value_mean = np.mean(max_each_mic)
max_value_median = np.median(np.sort(max_each_mic))
print('Max value:', max_value)
print('Mean of maximum value of all microphones',max_value_mean)
print('Median of maximum value of all microphones',max_value_median)
max_value = max_value_median*1.5 # use the median of the max value of each microphone as an amplitude limit for plots

# Find mics giving higher values than normal
high_val_mics = np.where(np.argmax(np.abs(data) > max_value_median*2, axis = 0)>0)[0] # all microphones with values over max_value_median*2

# manually selected bad mics where the signals should be set to zero
additional_mics = np.array([107])
delete_mics = np.hstack((high_val_mics, additional_mics))
delete_mics = np.sort(delete_mics)

# collect all microphones that should be ignored in the beamforming algorithm
ignored_mics = np.append(delete_mics, zero_mics)
ignored_mics = np.sort(ignored_mics)
np.save('unused_mics', ignored_mics)        # save ignored microphones to .npy file, to be loaded into the beamformer program

print('ignored mics,', ignored_mics)
print('high val mics', high_val_mics)
print('zero mics,', zero_mics)

# delete data (set data to 0) for the specified microphones
data_ignored = delete_mic_data(data, ignored_mics)

# plot array setup
if plot_array_setup:
    plot_array(delete_mics)

if all_mics:
    #plot_all_mics(data, amp_lim=1, samp_lim=1)
    plot_all_mics(data_ignored, N_mics = 4*64, amp_lim=1, samp_lim=1)

if all_individual:
    plot_all_individual(data, 0, 0)
    plot_all_individual(data, 0, 1)
    plot_all_individual(data, 0, 2)
    plot_all_individual(data, 0, 3)

# plot selected microphones
if selected:
    plot_selected(data, high_val_mics)

# plot energy spectrum
if energy:
    plot_energy(data, high_val_mics)

# show all plots
if show_plots:
    plt.show()