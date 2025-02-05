import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

def read_kria_bin(fpath, type='full', plot_flag=False):
    ''' 
    Output:
        1. SpkSparse: sparse COO matrix from scipy.sparse 
            which has the rows as the repeated idx of neurons and the cols as the timestamps in ms,
            rec_duration should be in ms: each neuron is repeated according to his timestamp
        2. vector of integers numbers from 1 to lenght of recording
        
        ex: 
            SparseSpk.row   SparseSpk.col
            neuronID        ts
            [0              149
            0               15511
            1               34
            1               368
            1               5640...]
    '''
    def reshape_data(SpkSparse):
        ''' Reshape the sparse matrices in a nested list with 1024 positions inside which there are all the corresponding tstamps '''
        
        tstamps_per_neuron = []
        for neuron in range(SpkSparse.shape[0]):
            indices = np.where(SpkSparse.row == neuron) #returns a tuple containing one ndarray
            if len(indices[0]) >= 1:
                tstamps_per_neuron.append(SpkSparse.col[indices[0]]) #the [0] beacuse you have to access to the tuple
            else:
                tstamps_per_neuron.append(np.array([0]))
        return tstamps_per_neuron
    
    def format2bit(x):
        Nbit = 32
        N, L = x.shape
        y = np.zeros((Nbit * N, L), dtype=bool)
        for ww in range(N):
            for k in range(1, Nbit + 1):
                y[ww * Nbit - k, :] = np.bitwise_and(x[ww, :], 1 << (k - 1)) > 0
        return y

    if type not in ['full', 'sparse']:
        raise ValueError(f'{type} not recognized as file format option.')

    # start_time = time.time()

    with open(fpath, 'rb') as fid:
        if type == 'full':
            A = np.fromfile(fid, dtype=np.uint32).reshape((33, -1), order='F')
            ts = A[0, :]
            SpkSparse = coo_matrix(format2bit(A[1:, :]))
        elif type == 'sparse':
            A = np.fromfile(fid, dtype=np.uint32).reshape((2, -1), order='F')
            SpkSparse = coo_matrix((np.ones(A.shape[1]), (A[0, :], A[1, :])))
            ts = None

    # end_time = time.time()
    # print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    if SpkSparse.row.size == 0:
        return [1]


    if plot_flag:
        plt.spy(SpkSparse, markersize=1)
        plt.show()
    
    spikes = []
    tstamps_per_neuron = reshape_data(SpkSparse)
    spikes.append(tstamps_per_neuron)

    return spikes, len(ts)/1e3







        

    
