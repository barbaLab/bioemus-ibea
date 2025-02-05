import numpy as np
import scipy
from scipy.stats import gaussian_kde, entropy
from scipy.special import rel_entr
from scipy.ndimage import gaussian_filter1d
from analysis.spike_analysis import spike_analysis
from sklearn.metrics import mean_squared_error
from analysis.burst_analysis import burst_analysis
from IBEAforBioemus.read_binary import read_kria_bin
from configuration.gen_config import gen_config, NetwConfParams
import subprocess
import os
import sys
import io
import psutil

class saveIO(io.StringIO):
    def __init__(self, original_stdout):
        super().__init__()
        self.original_stdout = original_stdout

    def write(self, s):
        self.original_stdout.write(s)
        return super().write(s)

    def flush(self):
        self.original_stdout.flush()
        return super().flush()

def clear_files(mydir: str, extension: str):
    try:
        for f in os.listdir(mydir):
            if not f.endswith(extension):
                continue
            os.remove(os.path.join(mydir, f))
    except FileNotFoundError:
        print(f"File not found: {mydir}")
    except PermissionError:
        print(f"Permission denied: unable to delete {mydir}.")
    except Exception as e:
        print(f"Error deleting {mydir}: {e}")

def load_goal():
    ISI_dict = scipy.io.loadmat('/home/ubuntu/bioemus/IBEAforBioemus/Goals/ISI.mat')
    MFR_dict = scipy.io.loadmat('/home/ubuntu/bioemus/IBEAforBioemus/Goals/MFR.mat')
    # LvR_dict = scipy.io.loadmat('/home/ubuntu/bioemus/bioemus_new/IBEAforBioemus/Goals/lvr.mat')
    MBR_dict = scipy.io.loadmat('/home/ubuntu/bioemus/IBEAforBioemus/Goals//MBR.mat')
    # ISI_goal = np.sum(ISI_dict['ISI'], axis=0)
    # ISI_goal /=np.sum(ISI_goal)
    ISI_goal = ISI_dict['ISI']
    cum_sum = []
    for i, vector in enumerate(ISI_goal):
        cum_sum.append(np.cumsum(vector + np.finfo(float).eps))
    cum_sum = np.array(cum_sum)
    average = np.mean(cum_sum, axis = 0)
    ISI_goal = np.diff(average)
    ISI_goal = np.concatenate(([average[0]], ISI_goal))
    ISI_goal /= np.sum(ISI_goal)
    MFR_goal = MFR_dict['mfr']
    # LvR_goal = LvR_dict['lvr']
    MBR_goal = MBR_dict['mbr']
    goal = [MFR_goal[0], MBR_goal[0], ISI_goal]

    return goal

def compute_goal_distr(goal):
    eps = np.finfo(float).eps
    goal_distr = []
    x = [np.linspace(0, 100, 199), np.linspace(0, 100, 199), np.linspace(0, 2, 199)]
    for j in range(len(goal)):
        if j == 2:
            goal_distr.append(gaussian_filter1d(goal[j], sigma=1))
            goal_distr[j] += eps
            goal_distr[j] /= np.sum(goal_distr[j])
        else:
            kde = gaussian_kde(goal[j])
            # x[i] = np.linspace(goal[j].min(), goal[j].max(), 200)
            goal_distr.append(kde(x[j]))
            goal_distr[j] += eps
            goal_distr[j] /= np.sum(goal_distr[j])
    return np.array(goal_distr)

def compute_goal_median(goal):
    MFR_goal_median = np.nanmedian(goal[0])
    MBR_goal_median = np.nanmedian(goal[1])
    # lvr_goal_median = np.nanmedian(goal[2])
    ISI_goal_median = goal[2]
    
    return [MFR_goal_median, MBR_goal_median, ISI_goal_median]

def do_analysis_median(filename, RASTER_LIST, parameters):

    result = read_kria_bin(filename, 'full', plot_flag = False) # Converting from .bin to list of neurons' timestamp
    if len(result) == 1:
        print(f"\tThe configuration {RASTER_LIST} with {parameters} produced no spikes, setting the metrices to unreal numbers")
        individual_metrics = [np.float64(0), np.float64(0), np.zeros(199)]
        return individual_metrics
    else:
        tstamp, REC_DURATION_S = result

    active_neurons = 0 
    for spike_train in tstamp[0]:
        if len(spike_train) > 2:
            active_neurons+=1

    if active_neurons <= 850: # at least 700 active neurons
        individual_metrics = [np.float64(0), np.float64(0), np.zeros(199)]
        return individual_metrics
    
    MFR, LVR, ISI = spike_analysis(RASTER_LIST, tstamp, REC_DURATION_S)
    MBR, IBI_global, length_burst_global = burst_analysis(RASTER_LIST, tstamp, REC_DURATION_S)

    MFR = np.nanmedian(MFR)
    # lvr = np.nanmedian(LVR)
    MBR = np.nanmedian(MBR)
    # if np.isnan([lvr]):
    #     lvr = np.float64(0)  
    # individual_metrics = [MFR, MBR, lvr, np.array(ISI)]
    individual_metrics = [MFR, MBR, np.array(ISI)]
    return individual_metrics

def do_analysis(filename, RASTER_LIST, parameters):

    result = read_kria_bin(filename, 'full', plot_flag = False) # Converting from .bin to list of neurons' timestamp
    if len(result) == 1:
        print(f"\tThe configuration {RASTER_LIST} with {parameters} produced no spikes, setting the metrices to unreal numbers")
        individual_metrics = [np.zeros(199), np.zeros(199), np.zeros(199)]
        return individual_metrics
    else:
        tstamp, REC_DURATION_S = result
    
    # Copute number of active neuron (at least 2 spikes)
    active_neurons = 0 
    for spike_train in tstamp[0]:
        if len(spike_train) > 2:
            active_neurons+=1

    if active_neurons <= 850: # at least 850 active neurons
        individual_metrics = [np.zeros(199), np.zeros(199), np.zeros(199)]
        return individual_metrics
    
    MFR, LVR, ISI = spike_analysis(RASTER_LIST, tstamp, REC_DURATION_S)
    MBR, IBI_global, length_burst_global = burst_analysis(RASTER_LIST, tstamp, REC_DURATION_S)

    individual_metrics = [np.array(MFR[0]), np.array(MBR[0]), np.array(ISI)]

    return individual_metrics

def compute_objectives(individual_metrics, goal_distr):
    
    def compute_KL_div(goal_distr, individual_metrics):

        eps = np.finfo(float).eps
        x1 = [np.linspace(0, 100, 199), np.linspace(0, 100, 199), np.linspace(0, 2, 199)]
        distance = []
        true_distr = []
        for i in range(len(individual_metrics)):
            all_zeros = np.all(individual_metrics[i]==0)
            if all_zeros:
                distance.append(np.float64(20000))
                true_distr.append(np.zeros(199)) 
            else:
                if i == 2:   # ISI case is already a distribution
                    true_distr.append(gaussian_filter1d(individual_metrics[i], sigma=1)) #smoothing ISI
                    true_distr[i] += eps 
                    true_distr[i] /= np.sum(true_distr[i]) 
                    distance.append(entropy(goal_distr[i], true_distr[i]))
                else:    
                    kde = gaussian_kde(individual_metrics[i]) # estimate of pdf
                    # x1 = np.linspace(goal[i].min(), goal[i].max(), 200)
                    true_distr.append(kde(x1[i])) # appending the pdf sampled over 200 samples
                    true_distr[i] += eps # to prevent division by 0 in entropy
                    true_distr[i] /= np.sum(true_distr[i]) #rinormalizing to mantain the same distribution
                    distance.append(entropy(goal_distr[i], true_distr[i])) # computing the kl divergency
        return np.array(distance), np.array(true_distr)
    
    
    if len(individual_metrics) != len(goal_distr):
        print("\tError: individuals and true values have different lengths")
        sys.exit()
    else:
        distance, true_distr = compute_KL_div(goal_distr, individual_metrics)

    distance[distance>3] = 3 

    return distance, true_distr


def compute_median_dist(individual_metrics, goal_distr):
    MFR_goal = goal_distr[0]
    MBR_goal = goal_distr[1]
    # LvR_goal = goal_distr[2]
    ISI_goal = goal_distr[2]

    std_MFR = 1
    std_MBR = 1
    # std_LvR = 1
        # dist = np.array([abs(individual_metrics[0]-MFR_goal)/std_MFR, abs(individual_metrics[1]-MBR_goal)/std_MBR,
        #                   abs(individual_metrics[2]-LvR_goal)/std_LvR, mean_squared_error(ISI_goal, individual_metrics[3], squared = False)])
    prova = [individual_metrics[0], individual_metrics[1], np.sum(individual_metrics[2])]
    all_zeros = np.any(np.array(prova)) == 0
    if all_zeros:
        dist = np.array([100, 100, 100])
    else:
        dist = np.array([abs(individual_metrics[0]-MFR_goal)/std_MFR, abs(individual_metrics[1]-MBR_goal)/std_MBR,
                            mean_squared_error(ISI_goal, individual_metrics[2], squared = False)])
    dist[dist>100] = 100 
    return dist, 0

def print_memory_info():
    # Get available memory
    mem = psutil.virtual_memory()
    available_memory = mem.available / (1024 * 1024)  # Convert bytes to MB
    print(f"\tAvailable memory: {available_memory:.2f} MB\n")


def evaluate_objectives_median(parameters, goal_medians, index, conf_name):

    '''
    Parameters must be a numpy array with 4 numbers: it contains the  parameters that are changed in each individual.
        In this case it changes the exc/inh ratio, the max probability of connection for the smallworld connectivity,
        the synaptic weight of the NMDA synapse, the ratio between AMPA/NMDA excitatory synapses and the synaptic noise.
    '''

    CONFIG_NAME = conf_name + index
    SAVE_PATH   = "config/"

    netw_conf = NetwConfParams()
    netw_conf.model                     = "ratbrain" # "custom", "single", "connectoid"
    netw_conf.emulation_time_s          = 5
    netw_conf.en_step_stim              = False
    netw_conf.step_stim_delay_ms        = 0
    netw_conf.step_stim_duration_ms     = 0
    netw_conf.local_save_path           = "/home/ubuntu/bioemus/data/"
    netw_conf.en_randomize_hh_params    = True
    netw_conf.val_randomize_hh_params   = 0.025 #0.025
    netw_conf.org_wsyninh               = 1.0
    netw_conf.org_wsynexc               = 0.22
    netw_conf.org_wnmda                 = parameters[3]
    netw_conf.org_wampa                 = parameters[4]
    netw_conf.org_pcon_in               = 0.08
    netw_conf.org_pcon_out              = 0.02
    netw_conf.org_pcon                  = parameters[1] # max probability of connection with smallworld
    netw_conf.org_wsyn_in               = 1.0
    netw_conf.org_wsyn_out              = 1.0
    netw_conf.org_inh_ratio             = parameters[0]
    netw_conf.org_exc_ratio             = parameters[2]
    netw_conf.org_inj_noise             = parameters[5]

    state = np.random.get_state()
    np.random.seed(333)
    [hwconfig, swconfig] = gen_config(CONFIG_NAME, netw_conf, SAVE_PATH)
    np.random.set_state(state)
    del(hwconfig)
    del(swconfig)

    password = 'kriachan'

    launch_name = f"echo {password} | sudo -S bash -c 'source init.sh && sudo -E ./launch_app.sh /home/ubuntu/bioemus/config/swconfig_" + CONFIG_NAME + ".json'"

    # Command to execute the .sh script
    result = subprocess.run([launch_name], shell=True, capture_output=True, text=True, executable='/bin/bash', timeout = netw_conf.emulation_time_s+30)

    print(f"\tConcluded emulation {CONFIG_NAME}, parameters: {parameters}")
    # print_memory_info()

    filename = "/home/ubuntu/bioemus/data/raster_" + CONFIG_NAME + ".bin"
    # Verifica se il file esiste
    if not(os.path.isfile(filename)):
        print(result)
        sys.exit(f"Error: file {filename} does not exist, maybe problem with the execution:\n")


    RASTER_LIST = CONFIG_NAME

    individual_metrics = do_analysis_median(filename, RASTER_LIST, parameters)

    distances, true_distributions,  = compute_median_dist(individual_metrics, goal_medians)

    return distances, true_distributions
    
def evaluate_objectives_DKL(parameters, goal_distr, index, conf_name):

    '''
    Parameters must be a numpy array with 4 numbers: it contains the  parameters that are changed in each individual.
        In this case it changes the exc/inh ratio, the max probability of connection for the smallworld connectivity,
        the synaptic weight of the NMDA synapse, the ratio between AMPA/NMDA excitatory synapses and the synaptic noise.
    '''

    CONFIG_NAME = conf_name + index
    SAVE_PATH   = "config/"

    netw_conf = NetwConfParams()
    netw_conf.model                     = "ratbrain" # "custom", "single", "connectoid"
    netw_conf.emulation_time_s          = 5
    netw_conf.en_step_stim              = False
    netw_conf.step_stim_delay_ms        = 0
    netw_conf.step_stim_duration_ms     = 0
    netw_conf.local_save_path           = "/home/ubuntu/bioemus/data/"
    netw_conf.en_randomize_hh_params    = True
    netw_conf.val_randomize_hh_params   = 0.025 #0.025
    netw_conf.org_wsyninh               = 1.0
    netw_conf.org_wsynexc               = 0.22
    netw_conf.org_wnmda                 = parameters[3]
    netw_conf.org_wampa                 = parameters[4]
    netw_conf.org_pcon_in               = 0.08
    netw_conf.org_pcon_out              = 0.02
    netw_conf.org_pcon                  = parameters[1] # max probability of connection with smallworld
    netw_conf.org_wsyn_in               = 1.0
    netw_conf.org_wsyn_out              = 1.0
    netw_conf.org_inh_ratio             = parameters[0]
    netw_conf.org_exc_ratio             = parameters[2]
    netw_conf.org_inj_noise             = parameters[5]

    state = np.random.get_state()
    np.random.seed(333)
    [hwconfig, swconfig] = gen_config(CONFIG_NAME, netw_conf, SAVE_PATH)
    np.random.set_state(state)
    del(hwconfig)
    del(swconfig)

    password = 'kriachan'

    launch_name = f"echo {password} | sudo -S bash -c 'source init.sh && sudo -E ./launch_app.sh /home/ubuntu/bioemus/config/swconfig_" + CONFIG_NAME + ".json'"

    # Command to execute the .sh script
    result = subprocess.run([launch_name], shell=True, capture_output=True, text=True, executable='/bin/bash', timeout = netw_conf.emulation_time_s+30)

    print(f"\tConcluded emulation {CONFIG_NAME}, parameters: {parameters}")
    # print_memory_info()

    filename = "/home/ubuntu/bioemus/data/raster_" + CONFIG_NAME + ".bin"
    # Verifica se il file esiste
    if not(os.path.isfile(filename)):
        print(result)
        sys.exit(f"Error: file {filename} does not exist, maybe problem with the execution:\n")


    RASTER_LIST = CONFIG_NAME

    individual_metrics = do_analysis(filename, RASTER_LIST, parameters)

    distances, true_distributions = compute_objectives(individual_metrics, goal_distr)

    return distances, true_distributions
