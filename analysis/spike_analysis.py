import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as mpl

def shape_data(data):
    
    timestamps      = []
    activeneurons   = []
    neuron          = data[:,1]
    neuron_list     = np.unique(data[:,1])
    
    for k in range(len(neuron_list)):
        
        index = np.where(neuron == neuron_list [k])[0]
        
        if len(index) >= 2 : # Consider only neurons that spiked more than once
            timestamps.append(data[index,0])
            activeneurons.append(neuron_list [k])
            
    return neuron_list, timestamps, activeneurons

def mean_firing_rate(events, rec_duration_ms) : 
    mean        = []
    spikecounts = []
    time        = rec_duration_ms # in [ms]
    '''calculate the mean firing rate of neuron signals'''

    for neuron in events : 
        spikecounts.append(len(neuron))
        mean.append(len(neuron)/(time/1e3)) # in [spikes/s]

    return mean, spikecounts

def LvR(t):
    """
    Returns the LvR for a given set of spike timestamps, t.
    Adapted from: 'Relating Neuronal Firing Patterns to Functional
    Differentiation of Cerebral Cortex.' Shinomoto et al. (2009)
    
    Parameters:
    t : numpy array
        1D array of spike timestamps (seconds)
    
    Returns:
    LvR_out : float
        LvR value for the given segment of a spike train.
    """
    
    # Parameters
    R = 0.0015  # Refractoriness (s)
    # Ensure t is a 1D array
    t = np.ravel(t)
    # Calculate interspike intervals
    I = np.diff(t)
    n = len(I)  # Number of interspike intervals
    if n < 10:  # Check if there are enough intervals to proceed
        return np.nan
    # Calculate the summand using vectorized operations
    I1 = I[:-1]
    I2 = I[1:]
    term1 = (1 - 4 * I1 * I2 / ((I1 + I2) ** 2))
    term2 = (1 + 4 * R / (I1 + I2))
    summand = np.sum(term1 * term2)
    # Normalization constant
    norm = 3 / (n - 1)
    # Output LvR value
    LvR_out = norm * summand 
    
    return LvR_out


def ISI_distribution(raster_list, events_list, rec_duration_ms) : 
    '''returns time interval between spikes histogram and boxplot of neuron signals'''
    
    # fig, ax     = plt.subplots(3, layout = 'tight', figsize = (15,15))
    # fig, ax     = plt.subplots(3, layout = 'tight', num="ISI + MFR")
    diff_total  = []
    mean_total  = []
    lvr_total = []
    # label       = raster_list
    # color_list = mpl.colormaps.get_cmap('tab10').resampled(len(raster_list)).colors
    for events in events_list :
        
        number_neurons = len(events)
        lvr = []
        diff = []
        mean_total.append(mean_firing_rate(events, rec_duration_ms)[0])
        for neuron in events :        
            if len(neuron) > 1 :     
                diff.append(neuron[1:]-neuron[:-1])
                lvr.append(LvR(neuron))
            else:
                lvr.append(np.nan)
                diff.append([np.nan])
        # diff = np.concatenate(diff, axis=0)
        diff_total.append(diff)
        lvr_total.append(lvr)
        
    tau = np.linspace(0,100, 200)  
    # tau = np.logspace(np.log10(min([min(i) for i in diff_total])),np.log10(max([max(i) for i in diff_total])), 200)
    # ISI_counts, bin_edges= np.histogram(diff, bins=tau, density=False)
    # ISI_counts /= np.sum(ISI_counts)
    cum_sum = []
    for i, vector in enumerate(diff):
        hist, bin_edg = np.histogram(vector, bins=tau, density=False)
        hist = hist.astype(float)
        hist /= np.sum(hist+ np.finfo(float).eps)
        cum_sum.append(np.cumsum(hist + np.finfo(float).eps))
    cum_sum = np.array(cum_sum)
    average = np.mean(cum_sum, axis = 0)
    ISI_counts = np.diff(average)
    ISI_counts = np.concatenate(([average[0]], ISI_counts))
    ISI_counts /= np.sum(ISI_counts)
    # if np.sum(ISI_counts) == 0:
    #     mean_total = lvr = ISI_counts = np.zeros(199)
    # else:
    #     ISI_counts = ISI_counts / np.sum(ISI_counts) 
    # ax[0].hist(diff_total, tau, color = color_list, edgecolor =  'k'  , stacked=False, label = label)
    
    # ax[0].set_xlabel(r'$\tau$-[ms]')
    # ax[0].set_ylabel('number')
    # ax[0].set_xscale('log')
    # ax[0].legend()
    # ax[0].set_title('Histogram of ISI distribution')
    
    # bp = ax[1].boxplot(diff_total,patch_artist=True, vert=True, meanline = True, showfliers= False, labels = label)
    # for patch, color in zip(bp['boxes'], color_list): 
    #     patch.set_facecolor(color)
    
    # ax[1].set_ylabel('ISI [ms]')
    # ax[1].set_title('Box plot ISI')
    
    # bp2 = ax[2].boxplot(mean_total,patch_artist=True, vert=True, meanline = True, showfliers= False, labels = label)
    # for patch, color in zip(bp2['boxes'], color_list): 
    #     patch.set_facecolor(color)
    
    # ax[2].set_ylabel('MFR [spikes/s]')
    # ax[2].set_title('Box plot MFR')
    
    return mean_total, lvr_total, ISI_counts

def spike_analysis(raster_list, tstamp_list, rec_duration_s):
    MFR, lvr, ISI_global = ISI_distribution(raster_list, tstamp_list, rec_duration_s*1e3)
    return MFR, lvr, ISI_global
