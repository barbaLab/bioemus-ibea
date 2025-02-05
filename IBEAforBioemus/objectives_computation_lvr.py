import numpy as np
from scipy.stats import gaussian_kde, entropy
from scipy.ndimage import gaussian_filter1d
from analysis.spike_analysis import spike_analysis
from analysis.burst_analysis import burst_analysis
from IBEAforBioemus.read_binary import read_kria_bin
from configuration.gen_config import gen_config, NetwConfParams
import subprocess
import os
import sys
import psutil

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

def compute_goal_distr(goal):
    eps = np.finfo(float).eps
    goal_distr = []
    x = [np.linspace(0, 100, 199), np.linspace(0, 100, 199), np.linspace(0, 2, 199)]
    for j in range(len(goal)):
        # if j == 3:
        #     goal_distr.append(gaussian_filter1d(goal[j], sigma=1))
        #     goal_distr[j] += eps
        #     goal_distr[j] /= np.sum(goal_distr[j])
        # else:
        kde = gaussian_kde(goal[j])
        # x[i] = np.linspace(goal[j].min(), goal[j].max(), 200)
        goal_distr.append(kde(x[j]))
        goal_distr[j] += eps
        goal_distr[j] /= np.sum(goal_distr[j])
    return np.array(goal_distr)

def do_analysis(filename, RASTER_LIST, parameters):

    result = read_kria_bin(filename, 'full', plot_flag = False) # Converting from .bin to list of neurons' timestamp
    if len(result) == 1:
        print(f"\tThe configuration {RASTER_LIST} with {parameters} produced no spikes, setting the metrices to unreal numbers")
        individual_metrics = [np.zeros(199), np.zeros(199), np.zeros(199), np.zeros(199)]
        return individual_metrics
    else:
        tstamp, REC_DURATION_S = result
    # Control that at least one neuron has at least 2 spikes (to calculate ISI)
    tot_spikes = 0
    for times in tstamp[0]:
        tot_spikes += len(times)
    if tot_spikes <= len(tstamp[0]):
        individual_metrics = [np.zeros(199), np.zeros(199), np.zeros(199)]
        return individual_metrics
    MFR, LVR, ISI = spike_analysis(RASTER_LIST, tstamp, REC_DURATION_S)
    MBR, IBI_global, length_burst_global = burst_analysis(RASTER_LIST, tstamp, REC_DURATION_S)


    LVR = np.array(LVR[0])
    lvr = LVR[~np.isnan(LVR)]
    if len(lvr)<200:
        lvr = np.zeros(199) 

    individual_metrics = [np.array(MFR[0]), np.array(MBR[0]),  lvr]


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
                # if i == 2:   # ISI case is already a distribution
                #     true_distr.append(gaussian_filter1d(individual_metrics[i], sigma=1)) #smoothing ISI
                #     true_distr[i] += eps 
                #     true_distr[i] /= np.sum(true_distr[i]) 
                #     distance.append(np.sum(rel_entr(goal_distr[i], individual_metrics[i])))
                # else:    
                kde = gaussian_kde(individual_metrics[i]) # estimate of pdf
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


    
def print_memory_info():
    # Get available memory
    mem = psutil.virtual_memory()
    available_memory = mem.available / (1024 * 1024)  # Convert bytes to MB
    print(f"\tAvailable memory: {available_memory:.2f} MB\n")
    
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
