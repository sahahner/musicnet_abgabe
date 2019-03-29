import os,sys,shutil,mmap
import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
from sklearn.metrics import average_precision_score

import time

import tensorflow as tf
import mir_eval
import config

sz_float = 4

training_mean = 0
training_std = 1

# ids that will be deleted from trianing set in case debugging mode is chosen
delete_ids = [2560, 1792, 2562, 2506, 2564, 2566, 1793, 2568, 2570, 2571, 2572, 2573, 2575, 2576, 2302, 2581, 2582, 
              2586, 2075, 2076, 2077, 2590, 2079, 2080, 2081, 2594, 2595, 2596, 2603, 2678, 2397, 2608, 2611, 2614, 
              2104, 2105, 2619, 2620, 2621, 2622, 2112, 2113, 2114, 2627, 2116, 2629, 2118, 2119, 2632, 2633, 2127, 
              2131, 2138, 2319, 2140, 2405, 2659, 2148, 2149, 2150, 2151, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 
              2161, 2677, 2166, 2167, 2168, 2169, 2239, 2325, 2177, 2178, 2179, 2180, 1729, 2186, 2242, 2194, 2195, 
              2196, 1829, 2198, 2200, 2201, 2202, 2203, 2204, 2330, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 
              2215, 2588, 2218, 2219, 2220, 2221, 2222, 2224, 2225, 2227, 2228, 2229, 2230, 2231, 2232, 2234, 2591, 
              2237, 2238, 1727, 1728, 2241, 1730, 2243, 2244, 1733, 1734, 2593, 2248, 1739, 2240, 2082, 1742, 2083, 
              1749, 1750, 1751, 1752, 1755, 1756, 1757, 1758, 1760, 1763, 1764, 1765, 1766, 1768, 2282, 1771, 1772, 
              1773, 1775, 1776, 1777, 2292, 2293, 2294, 2295, 2296, 2297, 1919, 1788, 1789, 1790, 2304, 2305, 2307, 
              2308, 2310, 2313, 2314, 2315, 1805, 2318, 1807, 2320, 2322, 1811, 1812, 1813, 1817, 1818, 2334, 2335, 
              2336, 1828, 2341, 2342, 2343, 2567, 2345, 2346, 1835, 2348, 2350, 2491, 2357, 2358, 2359, 2364, 2365, 
              2366, 2368, 2371, 2372, 2373, 2374, 2529, 2376, 2377, 2379, 2443, 2381, 2383, 2384, 1873, 2388, 2389, 
              2390, 2391, 2392, 2393, 2618, 2398, 2403, 2404, 1893, 2406, 2410, 2411, 2415, 2417, 2147, 2451, 2420, 
              2422, 2423, 2424, 1916, 2538, 1918, 2431, 2432, 2433, 1922, 1923, 2436, 2441, 2442, 1931, 1932, 2626, 
              1735, 2285, 1859, 2247, 2117, 2288, 2466, 2444, 2289, 2472, 2473, 2476, 2477, 2478, 2480, 2481, 2482, 
              2483, 2462, 2486, 2487, 2488, 2490, 2463, 2492, 2494, 2555, 2497, 1822, 2501, 2502, 2504, 2505, 2078, 2507]


def test_import(name):
    print(name)

def mean_std(data, train_ids):
    len_total = 0
    for rec_id in train_ids:
        len_total += data[rec_id][1]
    print(len_total)
    training_mean = 0
    for rec_id in train_ids:
        factor = data[rec_id][1]/len_total
        XX = np.frombuffer(data[rec_id][0] , dtype=np.float32).copy()
        mean = np.mean(XX[20:],dtype=np.float64)
        training_mean += mean * factor

    print( 'Mean of training set:',training_mean)
    print()
    
    training_std = 0
    for rec_id in train_ids:
        factor = np.sqrt(data[rec_id][1]/len_total)
        XX = np.frombuffer(data[rec_id][0] , dtype=np.float32).copy()
        std = np.std( (XX[20:] - training_mean ),dtype=np.float64)
        training_std += factor * std  

    print('Standard deviation:',std)
    
    return training_mean, training_std

##############################################################################################
# Data Preprocessing Functions

# function to obtain one data sample from recordings
# input:
# - model_stats: class of type model_results containing basic parameters (data, labels, mm)
# - rec_id: the id of the recording
# - timesteps: uneven integer. int(timesteps/2) gives the number of windows in each direction that are considered
# - s: frame-number to extract (starting point)
# - window: windowsize
# - pitchshift: shift the label by pitchshift halftones
# - scaling_factor: scaling factor in frequency domain (includes corrections because of pitch shift and jitter)
# - normalize: boolean (does not have effect any more)
# output
# - x: adjacent audio segments of shape (timesteps, window)
# - y: corresponsing labels, 0-1-vector of shape (timesteps, mm)
def get_data(model_stats,rec_id,timesteps,s,window,pitch_shift,scaling_factor,normalize=True):
   
    x = np.frombuffer(model_stats.data[rec_id][0][s*sz_float:int(
	s+scaling_factor*window*timesteps)*sz_float], dtype=np.float32).copy()

    #x = (x-training_mean)/training_std

    xp = np.arange(window*timesteps,dtype=np.float32)
    # interpolate the discrete function (np.arange(len(x)), x) at the points (scaling_factor*xp)
    x = np.interp(scaling_factor*xp,np.arange(len(x),dtype=np.float32),x).astype(np.float32)
    # real-valued audio frame with values in the range [−1,1], sampled at 44.1 kHz
    # -> x has length window

    x = np.reshape(x,(timesteps,window))

    for tt in range(timesteps):
        x[tt] /= np.linalg.norm(x[tt]) + config.epsilon

    y = np.zeros((timesteps,model_stats.mm))
    # select center of frame
    for tt in range(timesteps):
        for label in model_stats.labels[rec_id][s + (tt+1/2)*scaling_factor*window]:
            y[tt, label.data[1]+pitch_shift-model_stats.base_note] = 1 # apply pitch shift here (in frequency domain)
    
    return x,y

# function to create an input batch for one training step
# input:
# - model_stats: model_results-class containing training_ids
# - batch_size (integer)
# - timesteps: uneven integer. int(timesteps/2) gives the number of windows in each direction that are considered
# - window: window-length
# - pitch_transforms: integer in [-5,5], number of maximum semitones shifted
# - jitter: float in [-0.1,0.1], tuning variation
# output:
# - xmb: batch for a training step, size: (batch_size,timesteps,1,window,1)
# - ymb: corresponding labels to the batch xmb, size: (batch_size,timesteps,1,mm)
def get_training_batch(model_stats, batch_size, timesteps, window, pitch_transforms=0, jitter=0, normalize=True):
    if timesteps % 2 == 0:
        print("ERROR: Bidirectional memory cell is used. Uneven number of timesteps necessary.")
        print("Number of timesteps is augmented by one.")
        timesteps += 1
    
    # arrays for input data (batch)
    xmbatch = np.empty([batch_size,timesteps,1,window,1],dtype=np.float32)
    ymbatch = np.empty([batch_size,timesteps,1,model_stats.mm],dtype=np.float32)
        
    # picking randomly batch_size recordings
    rec_ids = [model_stats.train_ids[r] for r in \
               np.random.randint(0,len(model_stats.train_ids),batch_size)]
        
    if pitch_transforms > 0:
        transform_type = np.random.randint(-pitch_transforms,
                                           pitch_transforms,batch_size)

    if jitter > 0:
        jitter_amount = np.random.uniform(-jitter,jitter,batch_size)

    for i in range(batch_size):
        scaling_factor = 1
        shift = 0
        if pitch_transforms > 0:
            shift = transform_type[i]
            # half-step has a frequency ratio of 2^(1/12)
            # (shift + jitter_amount) halfsteps are shifted
            if jitter > 0:
                scaling_factor = (2.**((shift+jitter_amount[i])/12.))
            else:
                # only shift the halftones from pitch_transform
                scaling_factor = (2.**(shift/12.))
        
        # multiply in frequency domain by scaling_factor
        
        # pick a frame for training randomly
        s = np.random.randint(0, model_stats.data[rec_ids[i]][1] - timesteps * scaling_factor * window)
        xmbatch[i,:,0,:,0],ymbatch[i,:,0,:] = get_data(model_stats, rec_ids[i],timesteps,s,window,
                                                   pitch_shift=shift,scaling_factor=scaling_factor, normalize=normalize)
        
    return xmbatch, ymbatch


# Function to obtain training sample which is already saved in 
# numpy format in file. Normalization is already applied. 
# In case the numpy file does not exist and function is not run in debugging mode, 
# the numpy file is created.
# input:
# - model_stats: class of type model_results containing basic parameters (data, labels, mm)
# - timesteps: number of timesteps for each sequence
# - d: window size (integer)
# - normalize: boolean
# - debugging: boolean
# output:
# - Xtrain: training sample of size (nn, timesteps, 1, d, 1)
# - Ytrain: corresponding labels of size (nn, 128)
def get_training_sample(model_stats, timesteps, d=16384, normalize=True, debugging=False):
    rec_ids = model_stats.train_ids
    window = d
    cachex = config.tmp + 'Xtrain_ext_{}_{}_prep2.npy'.format(d,timesteps)
    cachey = config.tmp + 'Ytrain_ext_{}_{}_prep2.npy'.format(d,timesteps)
    #shape Xtrain: (32000, timesteps, 1, 16384, 1)
    #shape Ytrain: (32000, 128)
    if os.path.exists(cachex) and os.path.exists(cachey):
        Xtrain = np.load(cachex)
        Ytrain = np.load(cachey)
    else:       
        start_time = time.time()
        nn_seq = []
        # skip in beginning of recording
        offset = 44100
        
        #for every recording have at most 100 samples
        for i in range(len(rec_ids)):
            nn_seq += [min(int((model_stats.data[rec_ids[i]][1] - offset - (timesteps*window))/window),100)]
        
        print(sum(nn_seq), "training samples of", timesteps," timesteps to calculate training error")
        Xtrain = np.zeros([sum(nn_seq),timesteps,1,window,1])
        Ytrain = np.zeros([sum(nn_seq),model_stats.mm])
        index = 0
        for i in range(len(rec_ids)):
            count = nn_seq[i]
            X = np.zeros([count,timesteps,1,window,1])
            Y = np.zeros([count,model_stats.mm])
            # calculate the stride so that it matches the number of wanted samples
            stride = int((model_stats.data[rec_ids[i]][1] - offset - (timesteps*window))/count) 
            
            for j in range(count):
                X[j,:,0,:,0], Y_tmp = get_data(model_stats,rec_ids[i],timesteps,int(offset+j*stride),window,
                                           pitch_shift=0,scaling_factor=1, normalize=True)
                Y[j] = Y_tmp[int(timesteps/2)]

            Xtrain[index:index+count] = X
            Ytrain[index:index+count] = Y
                
        print('generated in {} seconds'.format(time.time()-start_time))
            
        if debugging == False:
            print("save xtrain to", cachex)
            print("size:",np.shape(Xtrain))
            np.save(cachex,Xtrain)
            print("save ytrain to", cachey)
            print("size:",np.shape(Ytrain))
            np.save(cachey,Ytrain)

    return Xtrain,Ytrain


# function to obtain test sample using get_data(). every sample is split 
# up in count sequences.
# input:
# - rec_ids: ids of test samples
# - timesteps: number of sequences to be included in each sample (for rnn)
# - window: size of the input frame
# - count: number of sequences to be created for each sample
# - fixed_stride: -1 (stride is adapted to sample) or nonnegative integer
# - pitch_shift: value by which recording is stretched or shrinked
# - normalize: boolean
# - debugging: boolean (without influence)
# output:
# - Xtest: test sample of size (count*len(rec_ids),timesteps,1,window,1)
# - Ytest: corresponding labels of size (count*len(rec_ids),mm) (only middle of sequence)
def get_test_sample(model_stats, timesteps, window, count, fixed_stride=-1, pitch_shift=0, normalize=True, debugging=False):
    
    rec_ids = model_stats.test_ids

    start_time=time.time()
    
    cachex = config.tmp + 'Xtest_ext_{}_{}_prep2.npy'.format(window,timesteps)
    cachey = config.tmp + 'Ytest_ext_{}_{}_prep2.npy'.format(window,timesteps)
        
    if os.path.exists(cachex) and os.path.exists(cachey):
        Xtest = np.load(cachex)
        Ytest = np.load(cachey)  
         
    else:
        # get number of seqeunces
        nn_seq = len(rec_ids) * count
        Xtest = np.zeros([nn_seq,timesteps,1,window,1])
        Ytest = np.zeros([nn_seq,model_stats.mm])
        # skip in beginning of recording
        offset = 44100
        # calculate scaling factor
        sf = 2.**(pitch_shift/12.)
        # for every sample create 1000 sequences
        for i in range(len(rec_ids)):
            X = np.zeros([count,timesteps,1,window,1])
            Y = np.zeros([count,model_stats.mm])
            # calculate the stride in case it is not fixed
            stride = (model_stats.data[rec_ids[i]][1] - offset - (timesteps*sf*window))/count if fixed_stride==-1 else fixed_stride
            for j in range(count):
                X[j,:,0,:,0], Y_tmp = get_data(model_stats, rec_ids[i],timesteps,int(offset+j*stride),window,
                                           pitch_shift=int(round(pitch_shift)),scaling_factor=sf)
                Y[j] = Y_tmp[int(timesteps/2)]

            Xtest[i*count:(i+1)*count] = X
            Ytest[i*count:(i+1)*count] = Y
              
        print('generated in {} seconds'.format(time.time()-start_time))
            
        print("save xtest to", cachex)
        print("size:",np.shape(Xtest))
        np.save(cachex,Xtest)
        print("save ytest to", cachey)
        print("size:",np.shape(Ytest))
        np.save(cachey,Ytest)
        
    return Xtest, Ytest

# Function to obtain sample of given rec_id to predict notes played at given timesteps
# input:
# - model_stats: class of type model_results containing basic parameters (data, labels, mm)
# - rec_id: id of the recording
# - timesteps: number of timesteps for each sequence (length nn)
# - window: window size (integer)
# - pitch_shift
# - normalize: boolean
# output:
# - Xtest: training sample of size (nn, timesteps, 1, window, 1)
# - Ytest: corresponding labels of size (nn, 128)
def get_times(model_stats, rec_id, times, timesteps, window, pitch_shift=0, normalize=True):
    # get number of seqeunces
    nn_seq = len(times)
    Xtest = np.zeros([nn_seq,timesteps,1,window,1])
    Ytest = np.zeros([nn_seq,model_stats.mm])
    # calculate scaling factor
    sf = 2.**(pitch_shift/12.)
    for j in range(nn_seq):
        s = int(times[j] - 0.5 * window*timesteps)
        Xtest[j,:,0,:,0], Y_tmp = get_data(model_stats,rec_id,timesteps,s,window,
                                   pitch_shift=int(round(pitch_shift)),scaling_factor=sf)
        Ytest[j] = Y_tmp[int(timesteps/2)]
    return Xtest, Ytest

##############################################################################################
# Graph Functions

# function to obtain filterbank  
# input:
# - d: length of receptive field of th created filters
# - k: number of filers created
# output:
# - wsin: ,size: (1, d, 1, k)
# - wcos: ,size: (1, d, 1, k)
def create_filters(d,k):
    # create k filters
    # d is the length of the receptive field (input)
    x = np.linspace(0, 2*np.pi, d, endpoint=False)
    wsin = np.empty((1,d,1,k), dtype=np.float32)
    wcos = np.empty((1,d,1,k), dtype=np.float32)
    start_freq = 50.
    end_freq = 6000.
    num_cycles = start_freq*d/11025. # here frequency needs to be changed                                         
    scaling_ind = np.log(end_freq/start_freq)/k
    window_mask = 1.0-1.0*np.cos(x)
    for ind in range(k):
        wsin[0,:,0,ind] = window_mask*np.sin(np.exp(ind*scaling_ind)*num_cycles*x)
        wcos[0,:,0,ind] = window_mask*np.cos(np.exp(ind*scaling_ind)*num_cycles*x)
    return wsin,wcos


# function to calculate the sizes of regions at layers of feed forward layer  
# input:
# - window: window size layer one
# - dd: receptive field layer one
# - stride: layer one stride
# - kk: input size for layer 2
# - d2_x, d2_y: : Parameters layer 2
# - stride_y: layer two stride
# - d3_x, d3_y: Parameters layer 3
# output:
# - num_regions: number of regions layer one
# - num_regions2_x, num_regions2_y: number of regions (x,y-dimension) layer two
# - num_regions3_x, num_regions3_y: number of regions (x,y-dimension) layer three
def calc_region_sizes(window, dd, stride, kk, d2_x, d2_y, stride_y, d3_x, d3_y):
    num_regions  = int(1 + (window-dd)/stride)
    print ('First layer regions: ({},{})'.format(num_regions,kk))
    num_regions2_x  = int(1 + (num_regions-d2_x)/1)
    num_regions2_y = int(1 + (kk-d2_y)/stride_y)
    print ('Second layer regions: ({},{})'.format(num_regions2_x,num_regions2_y))
    num_regions3_x = int(1 + (num_regions2_x - d3_x)/1)
    num_regions3_y = int(1 + (num_regions2_y - d3_y)/1)
    print ('Third layer regions: ({},{})'.format(num_regions3_x,num_regions3_y))
    return num_regions, num_regions2_x, num_regions2_y, num_regions3_x, num_regions3_y


# function for prediction without training that splits up the whole set
# in small subsets to prevent errors of type "ResourceExhaustedError"
# input: 
# - sess: the actual session
# - y_pd, direct_loss, xd, yd: nodes from neural network 
#   (predicted y, loss of prediction, input X, input Y)
# - X: sample, size: (n,1,window,1)
# - Y: corresponding labels, size: (n,mm)
# output:
#  Yhat: prediction, size: (n,mm)
#  mse: Mean Squared Error when comparing Y and Yhat
def predict_direct_model(sess, y_pd, direct_loss, xd, yd, X, Y, mm=128):
    Yhat = np.empty((len(X),mm))
    subdiv = 1000
    subset = X.shape[0]/subdiv
    # batches of size 11
    mse = 0
    for j in range(subdiv):
        #print(j)
        #print(np.shape(Y[int(subset*j):int(subset*(j+1))]))
        #print("run session")
        Yhat[int(subset*j):int(subset*(j+1))], se = sess.run([y_pd,direct_loss], 
                                                         feed_dict={xd: X[int(subset*j):int(subset*(j+1))],
                                                                    yd: Y[int(subset*j):int(subset*(j+1))]})
        mse += se
    return Yhat, mse/float(subdiv)


# define the class containing the results from the model and basic parameters
class model_results():
    def __init__(self, test_ids, train_ids, labels, data, base_note, mm):
        # weights
        self.weights = dict()
        # step
        self.iter = 0
        # result statistics
        self.stats = dict()
        self.stats['iter'] = [False,'{:<8}',[]]
        # time that has passed since last record
        self.stats['time'] = [True,'{:<8.0f}',[]]
        self.stats['lr'] = [False,'{:<8.6f}',[]]
        # mean squared errors
        self.stats['mse_train_1'] = [True,'{:<16.6f}',[]]
        self.stats['mse_train'] = [True,'{:<16.6f}',[]]
        self.stats['mse_test'] = [True,'{:<16.6f}',[]]
        # precicions
        self.stats['avp_train'] = [True,'{:<16.6f}',[]]
        self.stats['avp_test'] = [True,'{:<16.6f}',[]]
        
        self.averages = []
        
        self.test_ids = test_ids
        self.train_ids = train_ids
        self.labels = labels
        self.data = data
        self.base_note = base_note
        self.mm = mm

    # Register the weights inside the list w_list. 
    # The list name_list contains the corresponding names of the wariables in w_list
    def register_weights(self, w_list, name_list):
        if len(w_list) != len(name_list):
            print("error: input lists need to have same length")
        for ii in range(len(w_list)):
            w = w_list[ii]
            name = name_list[ii]
            # register the norm of the weights
            self.stats['n'+name] = [False,'{:<8.3f}',[]] 
            # register the weights
            self.weights[name] = w
        return None

    
###########################################################
# Graphs

# use 3-layer 11,025 Hz graph, after that an LSTM cell with 
def LSTM_graph_MomOpt_noReg(model_stats, batch_size, window, timesteps, stride, out, wsin, wcos, d2_x, d2_y, k2, stride_y, 
	d3_x, d3_y, k3, num_regions3_x, num_regions3_y, num_units, starter_learning_rate, decay_steps, lr_decay, mom=0.95, beta_reg=0):
	
    mm = model_stats.mm
    graph = tf.Graph()

    with graph.as_default():
        # if the input is a batch
        with tf.variable_scope('data_queue'):
            xb = tf.placeholder(tf.float32, shape=[None, timesteps, 1, window+(out-1)*stride, 1])
            yb = tf.placeholder(tf.float32, shape=[None, timesteps, out, mm])     
            # put batches and different timesteps in one dimension
            xq = tf.reshape(xb, (-1, 1, window+(out-1)*stride, 1))  
            # middle of sequence
            yq = yb[:,int(timesteps/2),:,:]

        # if the input is directly evaluated without training
        with tf.variable_scope('direct_data'):
            # batch and timestep dimension together for Convolution
            xd = tf.placeholder(tf.float32, shape=[None,timesteps,1,window,1]) 
            yd = tf.placeholder(tf.float32, shape=[None, mm])

        print ('---- Weights ----')
        with tf.variable_scope('parameters'): #glorot  
            w = tf.get_variable('w', [d2_x,d2_y,1,k2]) # 1 x 128 x 1 x 128 (16.384 values)
            print ('w',w)
            w2 = tf.get_variable('w2', [d3_x,d3_y,k2,k3]) # 13 x 1 x 128 x 1024 (1.703.936 values)
            print ('w2',w2)
            beta = tf.get_variable('beta', [int(num_regions3_x*num_regions3_y*k3),mm]) # (1*65*1024) x 128 (8.519.680 values)
            print ('beta',beta)
            w_proj = tf.get_variable('w_proj',[mm*2, mm]) #256 x 128
            print ('w_proj',w_proj)
            model_stats.register_weights([w,w2,beta,w_proj],['w','w2','beta','w_proj'])

        cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units, use_peepholes=True, state_is_tuple=True)
        #TODO: num_proj=mm and num_units bigger

        print ('\n---- Layers ----')
        with tf.variable_scope('model'):
            ##### Feed Forward Layers
            # data storage order: batch, height, width, channels
            zx = tf.square(tf.nn.conv2d(xq,wsin,strides=[1,1,stride,1],padding='VALID')) \
               + tf.square(tf.nn.conv2d(xq,wcos,strides=[1,1,stride,1],padding='VALID'))
            print ('zx',zx)
            # data storage order: batch, channels, height, width
            zx = tf.nn.relu(tf.nn.conv2d(tf.log(zx+10e-15),w,strides=[1,1,1,stride_y],padding='VALID',data_format='NCHW'))
            print ('z2',zx)
            zx = tf.nn.relu(tf.nn.conv2d(zx,w2,strides=[1,1,1,1],padding='VALID',data_format='NCHW'))
            print ('z3',zx)        
            zx = tf.matmul(tf.reshape(zx,[batch_size*timesteps,int(num_regions3_x*num_regions3_y*k3)]),beta)
            print ('y_feed_forward',zx)

            ##### Long short-term memory unit: Dynamic Bidirectional Recurrent 
            # separate batch and time dimension again
            zx = tf.reshape(zx, (-1, timesteps, mm))
            print('\n---- LSTM-Cell ----')
            print('Number of units:',num_units)
            print('Shape of input:',zx.get_shape()) # [batch_size, max_time, ...]

            outputs, states  = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell,
                cell_bw=cell,
                dtype=tf.float32,
                inputs=zx)
            #sequence_length=timesteps*np.ones(batch_size),

            output_fw, output_bw = outputs
            #states_fw, states_bw = states

            #compare only the outputs of middle of sequence
            output_fw = output_fw[:,int(timesteps/2),:]
            output_bw = output_bw[:,int(timesteps/2),:]
            print('Shape of output: 2 x',output_bw.get_shape()) # [batch_size, max_time, ...]

            ##### Feed Forward Layer: Project to final output dimension
            # project forward and backward output to final output
            y_p = tf.matmul(tf.concat([output_fw, output_bw], -1),w_proj)

            ##### Loss   
            loss = tf.reduce_mean(tf.nn.l2_loss(y_p-tf.reshape(yq,[batch_size,mm])))

        # learning rate decay
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, lr_decay, staircase=True)

        reg = tf.nn.l2_loss(w) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(beta) + tf.nn.l2_loss(w_proj)
        opt_op = tf.train.MomentumOptimizer(lr,mom).minimize(loss, global_step=global_step)

        # Create an ExponentialMovingAverage object
        ema = tf.train.ExponentialMovingAverage(decay=0.9998)

        with tf.control_dependencies([opt_op]):
            training_op = ema.apply([w, w2, beta, w_proj])

        with tf.variable_scope('model', reuse=True):
            # use ema.average(var) to return the average of variable var
            wavg = ema.average(w)
            w2avg = ema.average(w2)
            betaavg = ema.average(beta)
            w_projavg = ema.average(w_proj)

            xd_in = tf.reshape(xd, (-1, 1, window, 1))
            direct_zx = tf.square(tf.nn.conv2d(xd_in,wsin,strides=[1,1,stride,1],padding='VALID')) \
                      + tf.square(tf.nn.conv2d(xd_in,wcos,strides=[1,1,stride,1],padding='VALID'))
            direct_zx = tf.nn.relu(tf.nn.conv2d(tf.log(direct_zx+10e-15),wavg,strides=[1,1,1,stride_y],padding='VALID',data_format='NCHW'))
            direct_zx = tf.nn.relu(tf.nn.conv2d(direct_zx,w2avg,strides=[1,1,1,1],padding='VALID',data_format='NCHW'))
            direct_zx = tf.matmul(tf.reshape(direct_zx,[tf.shape(xd_in)[0],int(num_regions3_x*num_regions3_y*k3)]),betaavg)    
            direct_zx = tf.reshape(direct_zx, (-1, timesteps, mm))

            #apply lstm
            direct_outputs, direct_states  = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell,
                cell_bw=cell,
                dtype=tf.float32,
                inputs=direct_zx)

            direct_output_fw, direct_output_bw = direct_outputs
            #states_fw, states_bw = states

            #compare only the outputs of middle of sequence
            direct_output_fw = direct_output_fw[:,int(timesteps/2),:]
            direct_output_bw = direct_output_bw[:,int(timesteps/2),:]

            y_pd = tf.matmul(tf.concat([direct_output_fw, direct_output_bw], -1),w_projavg)
            direct_loss = tf.reduce_mean(tf.nn.l2_loss(y_pd-yd))

        # train saver
        saver = tf.train.Saver()
    
    return graph, training_op, loss, global_step, lr, reg, y_pd, direct_loss, xb, yb, xd, yd, saver


# use cross entropy
def LSTM_graph_AdamOpt_sig_noReg(model_stats, batch_size, window, timesteps, stride, out, wsin, wcos, d2_x, d2_y, k2, stride_y, 
	d3_x, d3_y, k3, num_regions3_x, num_regions3_y, num_units, starter_learning_rate, decay_steps, lr_decay, beta1=0.9, beta2=0.999, beta_reg=0):
	
	mm = model_stats.mm
	graph = tf.Graph()

	with graph.as_default():
		 
		# if the input is a batch
		with tf.variable_scope('data_queue'):
			xb = tf.placeholder(tf.float32, shape=[None, timesteps, 1, window+(out-1)*stride, 1])
			yb = tf.placeholder(tf.float32, shape=[None, timesteps, out, mm])	
			# put batches and different timesteps in one dimension
			xq = tf.reshape(xb, (-1, 1, window+(out-1)*stride, 1))  
			# middle of sequence
			yq = yb[:,int(timesteps/2),:,:]
			
		# if the input is directly evaluated without training
		with tf.variable_scope('direct_data'):
			# batch and timestep dimension together for Convolution
			xd = tf.placeholder(tf.float32, shape=[None,timesteps,1,window,1]) 
			yd = tf.placeholder(tf.float32, shape=[None, mm])
		
		print ('---- Weights ----')
		with tf.variable_scope('parameters'): #glorot  
			w = tf.get_variable('w', [d2_x,d2_y,1,k2]) # 1 x 128 x 1 x 128 (16.384 values)
			print ('w',w)
			w2 = tf.get_variable('w2', [d3_x,d3_y,k2,k3]) # 13 x 1 x 128 x 1024 (1.703.936 values)
			print ('w2',w2)
			beta = tf.get_variable('beta', [int(num_regions3_x*num_regions3_y*k3),mm]) # (1*65*1024) x 128 (8.519.680 values)
			print ('beta',beta)
			w_proj = tf.get_variable('w_proj',[mm*2, mm]) #256 x 128
			print ('w_proj',w_proj)
			model_stats.register_weights([w,w2,beta,w_proj],['w','w2','beta','w_proj'])

		cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units, use_peepholes=True, state_is_tuple=True)
		#TODO: num_proj=mm and num_units bigger
			
		print ('\n---- Layers ----')
		with tf.variable_scope('model'):
			##### Feed Forward Layers
			# data storage order: batch, height, width, channels
			zx = tf.square(tf.nn.conv2d(xq,wsin,strides=[1,1,stride,1],padding='VALID')) \
			   + tf.square(tf.nn.conv2d(xq,wcos,strides=[1,1,stride,1],padding='VALID'))
			print ('zx',zx)
			# data storage order: batch, channels, height, width
			zx = tf.nn.relu(tf.nn.conv2d(tf.log(zx+10e-15),w,strides=[1,1,1,stride_y],padding='VALID',data_format='NCHW'))
			print ('z2',zx)
			zx = tf.nn.relu(tf.nn.conv2d(zx,w2,strides=[1,1,1,1],padding='VALID',data_format='NCHW'))
			print ('z3',zx)			
			zx = tf.matmul(tf.reshape(zx,[batch_size*timesteps,int(num_regions3_x*num_regions3_y*k3)]),beta)
			print ('y_feed_forward',zx)
			
			##### Long short-term memory unit: Dynamic Bidirectional Recurrent 
			# separate batch and time dimension again
			zx = tf.reshape(zx, (-1, timesteps, mm))
			print('\n---- LSTM-Cell ----')
			print('Number of units:',num_units)
			print('Shape of input:',zx.get_shape()) # [batch_size, max_time, ...]
			outputs, states  = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=cell,
				cell_bw=cell,
				dtype=tf.float32,
				inputs=zx)
			#sequence_length=timesteps*np.ones(batch_size),
			output_fw, output_bw = outputs
			#compare only the outputs of middle of sequence
			output_fw = output_fw[:,int(timesteps/2),:]
			output_bw = output_bw[:,int(timesteps/2),:]
			print('Shape of output: 2 x',output_bw.get_shape()) # [batch_size, max_time, ...]
			
			##### Feed Forward Layer: Project to final output dimension
			# project forward and backward output to final output
			y_p = tf.sigmoid(tf.matmul(tf.concat([output_fw, output_bw], -1),w_proj))
	
			##### Loss
			loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(yq,[batch_size,mm]), logits=y_p) 
		
		# learning rate decay does not apply
		global_step = tf.Variable(0, trainable=False)
		#lr = starter_learning_rate
		lr = tf.train.exponential_decay(starter_learning_rate, global_step, 1000000, 0, staircase=True)
		reg = tf.nn.l2_loss(w) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(beta) + tf.nn.l2_loss(w_proj)
		opt_op = tf.train.AdamOptimizer(starter_learning_rate,beta1=beta1, beta2=beta2).minimize(loss, global_step=global_step)

		# Create an ExponentialMovingAverage object
		ema = tf.train.ExponentialMovingAverage(decay=0.9998)
		
		with tf.control_dependencies([opt_op]):
			training_op = ema.apply([w, w2, beta, w_proj])
		
		with tf.variable_scope('model', reuse=True):
			# use ema.average(var) to return the average of variable var
			wavg = ema.average(w)
			w2avg = ema.average(w2)
			betaavg = ema.average(beta)
			w_projavg = ema.average(w_proj)
			xd_in = tf.reshape(xd, (-1, 1, window, 1))
			direct_zx = tf.square(tf.nn.conv2d(xd_in,wsin,strides=[1,1,stride,1],padding='VALID')) \
					  + tf.square(tf.nn.conv2d(xd_in,wcos,strides=[1,1,stride,1],padding='VALID'))
			direct_zx = tf.nn.relu(tf.nn.conv2d(tf.log(direct_zx+10e-15),wavg,strides=[1,1,1,stride_y],padding='VALID',data_format='NCHW'))
			direct_zx = tf.nn.relu(tf.nn.conv2d(direct_zx,w2avg,strides=[1,1,1,1],padding='VALID',data_format='NCHW'))
			direct_zx = tf.matmul(tf.reshape(direct_zx,[tf.shape(xd_in)[0],int(num_regions3_x*num_regions3_y*k3)]),betaavg)		
			direct_zx = tf.reshape(direct_zx, (-1, timesteps, mm))
			#apply lstm
			direct_outputs, direct_states  = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=cell,
				cell_bw=cell,
				dtype=tf.float32,
				inputs=direct_zx)
			direct_output_fw, direct_output_bw = direct_outputs		
			#compare only the outputs of middle of sequence
			direct_output_fw = direct_output_fw[:,int(timesteps/2),:]
			direct_output_bw = direct_output_bw[:,int(timesteps/2),:]		
			y_pd = tf.sigmoid(tf.matmul(tf.concat([direct_output_fw, direct_output_bw], -1),w_projavg))				
			direct_loss = tf.reduce_mean(tf.nn.l2_loss(y_pd-yd))
			
		# train saver
		saver = tf.train.Saver()
	
	return graph, training_op, loss, global_step, lr, reg, y_pd, direct_loss, xb, yb, xd, yd, saver


# use 3-layer 11,025 Hz graph for comparison and for pretraining weights
def NoLSTM_graph_MomOpt_noReg(model_stats, batch_size, window, timesteps, stride, out, wsin, wcos, d2_x, d2_y, k2, stride_y, 
	d3_x, d3_y, k3, num_regions3_x, num_regions3_y, num_units, starter_learning_rate, decay_steps, lr_decay, mom=0.95, beta_reg=0):
	
	mm = model_stats.mm
	graph = tf.Graph()

	with graph.as_default():
	     
		# if the input is a batch
		with tf.variable_scope('data_queue'):
			xb = tf.placeholder(tf.float32, shape=[None, timesteps, 1, window+(out-1)*stride, 1])
			yb = tf.placeholder(tf.float32, shape=[None, timesteps, out, mm])
			# put batches and different timesteps in one dimension
			xq = tf.reshape(xb, (-1, 1, window+(out-1)*stride, 1))  
			# middle of sequence
			yq = yb[:,int(timesteps/2),:,:]
		
		# if the input is directly evaluated without training
		with tf.variable_scope('direct_data'):
			# batch and timestep dimension together for Convolution
			xd = tf.placeholder(tf.float32, shape=[None,timesteps,1,window,1]) 
			yd = tf.placeholder(tf.float32, shape=[None, mm])
		    
		print ('---- Weights ----')
		with tf.variable_scope('parameters'): #glorot  
			w = tf.get_variable('w', [d2_x,d2_y,1,k2]) # 1 x 128 x 1 x 128 (16.384 values)
			print ('w',w)
			w2 = tf.get_variable('w2', [d3_x,d3_y,k2,k3]) # 13 x 1 x 128 x 1024 (1.703.936 values)
			print ('w2',w2)
			beta = tf.get_variable('beta', [int(num_regions3_x*num_regions3_y*k3),mm]) # (1*65*1024) x 128 (8.519.680 values)
			print ('beta',beta)
			model_stats.register_weights([w,w2,beta],['w','w2','beta'])

		print ('\n---- Layers ----')
		with tf.variable_scope('model'):
			##### Feed Forward Layers
			# data storage order: batch, height, width, channels
			zx = tf.square(tf.nn.conv2d(xq,wsin,strides=[1,1,stride,1],padding='VALID')) \
			   + tf.square(tf.nn.conv2d(xq,wcos,strides=[1,1,stride,1],padding='VALID'))
			print ('zx',zx)
			# data storage order: batch, channels, height, width
			zx = tf.nn.relu(tf.nn.conv2d(tf.log(zx+10e-15),w,strides=[1,1,1,stride_y],padding='VALID',data_format='NCHW'))
			print ('z2',zx)
			zx = tf.nn.relu(tf.nn.conv2d(zx,w2,strides=[1,1,1,1],padding='VALID',data_format='NCHW'))
			print ('z3',zx)	
			y_p = tf.matmul(tf.reshape(zx,[batch_size*timesteps,int(num_regions3_x*num_regions3_y*k3)]),beta)
			print ('y_feed_forward',y_p)
				        
			##### Loss  
			loss = tf.reduce_mean(tf.nn.l2_loss(y_p-tf.reshape(yq,[batch_size,mm])))
		
		   
		# learning rate decay
		global_step = tf.Variable(0, trainable=False)
		lr = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.95, staircase=True)
		reg = tf.nn.l2_loss(w) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(beta)
		opt_op = tf.train.MomentumOptimizer(lr,mom).minimize(loss , global_step=global_step)
		# Create an ExponentialMovingAverage object
		ema = tf.train.ExponentialMovingAverage(decay=0.9998)

		with tf.control_dependencies([opt_op]):
			training_op = ema.apply([w, w2, beta])
		    
		with tf.variable_scope('model', reuse=True):
			# use ema.average(var) to return the average of variable var
			wavg = ema.average(w)
			w2avg = ema.average(w2)
			betaavg = ema.average(beta)

			xd_in = tf.reshape(xd, (-1, 1, window, 1))
			direct_zx = tf.square(tf.nn.conv2d(xd_in,wsin,strides=[1,1,stride,1],padding='VALID')) \
				  + tf.square(tf.nn.conv2d(xd_in,wcos,strides=[1,1,stride,1],padding='VALID'))
			direct_zx = tf.nn.relu(tf.nn.conv2d(tf.log(direct_zx+10e-15),wavg,strides=[1,1,1,stride_y],padding='VALID',data_format='NCHW'))
			direct_zx = tf.nn.relu(tf.nn.conv2d(direct_zx,w2avg,strides=[1,1,1,1],padding='VALID',data_format='NCHW'))
			y_pd = tf.matmul(tf.reshape(direct_zx,[tf.shape(xd_in)[0],int(num_regions3_x*num_regions3_y*k3)]),betaavg)        
			direct_loss = tf.reduce_mean(tf.nn.l2_loss(y_pd-yd))

		# train saver
		saver = tf.train.Saver({'w':w,'wavg':wavg,'w2':w2,'w2avg':w2avg,'beta':beta,'betaavg':betaavg})

	return graph, training_op, loss, global_step, lr, reg, y_pd, direct_loss, xb, yb, xd, yd, saver

# Graph using pretrained weights
def LSTM_PT_graph_MomOpt_noReg(model_stats, batch_size, window, timesteps, stride, out, wsin, wcos, d2_x, d2_y, k2, stride_y, 
	d3_x, d3_y, k3, num_regions3_x, num_regions3_y, num_units, starter_learning_rate, decay_steps, lr_decay, mom=0.95, beta_reg=0):
    
	mm = model_stats.mm
	graph = tf.Graph()

	with graph.as_default():
		# if the input is a batch
		with tf.variable_scope('data_queue'):
			xb = tf.placeholder(tf.float32, shape=[None, timesteps, 1, window+(out-1)*stride, 1])
			yb = tf.placeholder(tf.float32, shape=[None, timesteps, out, mm])
			# put batches and different timesteps in one dimension
			xq = tf.reshape(xb, (-1, 1, window+(out-1)*stride, 1))  
			# middle of sequence
			yq = yb[:,int(timesteps/2),:,:]
	
		# if the input is directly evaluated without training
		with tf.variable_scope('direct_data'):
			# batch and timestep dimension together for Convolution
			xd = tf.placeholder(tf.float32, shape=[None,timesteps,1,window,1]) 
			yd = tf.placeholder(tf.float32, shape=[None, mm])
	    
		print ('---- Weights ----')
		with tf.variable_scope('parameters'): #glorot  
			w = tf.get_variable('w', [d2_x,d2_y,1,k2], trainable=False) # 1 x 128 x 1 x 128 (16.384 values)
			print ('w',w)
			w2 = tf.get_variable('w2', [d3_x,d3_y,k2,k3], trainable=False) # 13 x 1 x 128 x 1024 (1.703.936 values)
			print ('w2',w2)
			beta = tf.get_variable('beta', [int(num_regions3_x*num_regions3_y*k3),mm], trainable=False) # (1*65*1024) x 128 (8.519.680 values)
			print ('beta',beta)
	
		with tf.variable_scope('parameters_train'): #glorot 
			w_proj = tf.get_variable('w_proj',[num_units*2, mm]) #256 x 128
			print ('w_proj',w_proj)
	
		model_stats.register_weights([w,w2,beta,w_proj],['w','w2','beta','w_proj'])

		cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units, use_peepholes=True, state_is_tuple=True)
		#TODO: num_proj=mm and num_units bigger

		print ('\n---- Layers ----')
		with tf.variable_scope('model'):
			##### Feed Forward Layers
			# data storage order: batch, height, width, channels
			zx = tf.square(tf.nn.conv2d(xq,wsin,strides=[1,1,stride,1],padding='VALID')) \
			   + tf.square(tf.nn.conv2d(xq,wcos,strides=[1,1,stride,1],padding='VALID'))
			print ('zx',zx)
			# data storage order: batch, channels, height, width
			zx = tf.nn.relu(tf.nn.conv2d(tf.log(zx+10e-15),w,strides=[1,1,1,stride_y],padding='VALID',data_format='NCHW'))
			print ('z2',zx)
			zx = tf.nn.relu(tf.nn.conv2d(zx,w2,strides=[1,1,1,1],padding='VALID',data_format='NCHW'))
			print ('z3',zx)
			zx = tf.matmul(tf.reshape(zx,[batch_size*timesteps,int(num_regions3_x*num_regions3_y*k3)]),beta)
			print ('y_feed_forward',zx)

			##### Long short-term memory unit: Dynamic Bidirectional Recurrent 
			# separate batch and time dimension again
			zx = tf.reshape(zx, (-1, timesteps, mm))
			print('\n---- LSTM-Cell ----')
			print('Number of units:',num_units)
			print('Shape of input:',zx.get_shape()) # [batch_size, max_time, ...]
			outputs, states  = tf.nn.bidirectional_dynamic_rnn(
			    cell_fw=cell,
			    cell_bw=cell,
			    dtype=tf.float32,
			    inputs=zx)
			output_fw, output_bw = outputs
			#compare only the outputs of middle of sequence
			output_fw = output_fw[:,int(timesteps/2),:]
			output_bw = output_bw[:,int(timesteps/2),:]
			print('Shape of output: 2 x',output_bw.get_shape()) # [batch_size, max_time, ...]

			##### Feed Forward Layer: Project to final output dimension
			# project forward and backward output to final output
			y_p = tf.matmul(tf.concat([output_fw, output_bw], -1),w_proj)
			print ('y_output:',y_p)
		
			##### Loss    
			loss = tf.reduce_mean(tf.nn.l2_loss(y_p-tf.reshape(yq,[batch_size,mm])))
		   
		# learning rate decay
		global_step = tf.Variable(0, trainable=False)
		lr = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.95, staircase=True)
		reg = tf.nn.l2_loss(w_proj)
		opt_op = tf.train.MomentumOptimizer(lr,mom).minimize(loss, global_step=global_step)

		# Create an ExponentialMovingAverage object
		ema = tf.train.ExponentialMovingAverage(decay=0.9998)

		with tf.control_dependencies([opt_op]):
			training_op = ema.apply([w, w2, beta, w_proj])

		with tf.variable_scope('model', reuse=True):
			# use ema.average(var) to return the average of variable var
			wavg = ema.average(w)
			w2avg = ema.average(w2)
			betaavg = ema.average(beta)
			w_projavg = ema.average(w_proj)

			xd_in = tf.reshape(xd, (-1, 1, window, 1))
			direct_zx = tf.square(tf.nn.conv2d(xd_in,wsin,strides=[1,1,stride,1],padding='VALID')) \
				  + tf.square(tf.nn.conv2d(xd_in,wcos,strides=[1,1,stride,1],padding='VALID'))
			direct_zx = tf.nn.relu(tf.nn.conv2d(tf.log(direct_zx+10e-15),wavg,strides=[1,1,1,stride_y],padding='VALID',data_format='NCHW'))
			direct_zx = tf.nn.relu(tf.nn.conv2d(direct_zx,w2avg,strides=[1,1,1,1],padding='VALID',data_format='NCHW'))
			direct_zx = tf.matmul(tf.reshape(direct_zx,[tf.shape(xd_in)[0],int(num_regions3_x*num_regions3_y*k3)]),betaavg)
			direct_zx = tf.reshape(direct_zx, (-1, timesteps, mm))
			#apply lstm
			direct_outputs, direct_states  = tf.nn.bidirectional_dynamic_rnn(
			    cell_fw=cell,
			    cell_bw=cell,
			    dtype=tf.float32,
			    inputs=direct_zx)
			direct_output_fw, direct_output_bw = direct_outputs
			#compare only the outputs of middle of sequence
			direct_output_fw = direct_output_fw[:,int(timesteps/2),:]
			direct_output_bw = direct_output_bw[:,int(timesteps/2),:]
			y_pd = tf.matmul(tf.concat([direct_output_fw, direct_output_bw], -1),w_projavg)
			direct_loss = tf.reduce_mean(tf.nn.l2_loss(y_pd-yd))

		# train saver
		saver = tf.train.Saver()

	return graph, training_op, loss, global_step, lr, reg, y_pd, direct_loss, xb, yb, xd, yd, saver, w, wavg, w2, w2avg, beta, betaavg

def LSTM_PT_graph_AdamOpt_noReg(model_stats, batch_size, window, timesteps, stride, out, wsin, wcos, d2_x, d2_y, k2, stride_y, 
	d3_x, d3_y, k3, num_regions3_x, num_regions3_y, num_units, starter_learning_rate, decay_steps, lr_decay, beta1=0.9, beta2=0.999, beta_reg=0):
    
	mm = model_stats.mm
	graph = tf.Graph()

	with graph.as_default():
		# if the input is a batch
		with tf.variable_scope('data_queue'):
			xb = tf.placeholder(tf.float32, shape=[None, timesteps, 1, window+(out-1)*stride, 1])
			yb = tf.placeholder(tf.float32, shape=[None, timesteps, out, mm])
			# put batches and different timesteps in one dimension
			xq = tf.reshape(xb, (-1, 1, window+(out-1)*stride, 1))  
			# middle of sequence
			yq = yb[:,int(timesteps/2),:,:]
	
		# if the input is directly evaluated without training
		with tf.variable_scope('direct_data'):
			# batch and timestep dimension together for Convolution
			xd = tf.placeholder(tf.float32, shape=[None,timesteps,1,window,1]) 
			yd = tf.placeholder(tf.float32, shape=[None, mm])
	    
		print ('---- Weights ----')
		with tf.variable_scope('parameters'): #glorot  
			w = tf.get_variable('w', [d2_x,d2_y,1,k2], trainable=False) # 1 x 128 x 1 x 128 (16.384 values)
			print ('w',w)
			w2 = tf.get_variable('w2', [d3_x,d3_y,k2,k3], trainable=False) # 13 x 1 x 128 x 1024 (1.703.936 values)
			print ('w2',w2)
			beta = tf.get_variable('beta', [int(num_regions3_x*num_regions3_y*k3),mm], trainable=False) # (1*65*1024) x 128 (8.519.680 values)
			print ('beta',beta)
	
		with tf.variable_scope('parameters_train'): #glorot 
			w_proj = tf.get_variable('w_proj',[num_units*2, mm]) #256 x 128
			print ('w_proj',w_proj)
	
		model_stats.register_weights([w,w2,beta,w_proj],['w','w2','beta','w_proj'])

		cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units, use_peepholes=True, state_is_tuple=True)
		#TODO: num_proj=mm and num_units bigger

		print ('\n---- Layers ----')
		with tf.variable_scope('model'):
			##### Feed Forward Layers
			# data storage order: batch, height, width, channels
			zx = tf.square(tf.nn.conv2d(xq,wsin,strides=[1,1,stride,1],padding='VALID')) \
			   + tf.square(tf.nn.conv2d(xq,wcos,strides=[1,1,stride,1],padding='VALID'))
			print ('zx',zx)
			# data storage order: batch, channels, height, width
			zx = tf.nn.relu(tf.nn.conv2d(tf.log(zx+10e-15),w,strides=[1,1,1,stride_y],padding='VALID',data_format='NCHW'))
			print ('z2',zx)
			zx = tf.nn.relu(tf.nn.conv2d(zx,w2,strides=[1,1,1,1],padding='VALID',data_format='NCHW'))
			print ('z3',zx)
			zx = tf.matmul(tf.reshape(zx,[batch_size*timesteps,int(num_regions3_x*num_regions3_y*k3)]),beta)
			print ('y_feed_forward',zx)

			##### Long short-term memory unit: Dynamic Bidirectional Recurrent 
			# separate batch and time dimension again
			zx = tf.reshape(zx, (-1, timesteps, mm))
			print('\n---- LSTM-Cell ----')
			print('Number of units:',num_units)
			print('Shape of input:',zx.get_shape()) # [batch_size, max_time, ...]
			outputs, states  = tf.nn.bidirectional_dynamic_rnn(
			    cell_fw=cell,
			    cell_bw=cell,
			    dtype=tf.float32,
			    inputs=zx)
			output_fw, output_bw = outputs
			#compare only the outputs of middle of sequence
			output_fw = output_fw[:,int(timesteps/2),:]
			output_bw = output_bw[:,int(timesteps/2),:]
			print('Shape of output: 2 x',output_bw.get_shape()) # [batch_size, max_time, ...]

			##### Feed Forward Layer: Project to final output dimension
			# project forward and backward output to final output
			y_p = tf.matmul(tf.concat([output_fw, output_bw], -1),w_proj)
			print ('y_output:',y_p)
		
			##### Loss    
			loss = tf.reduce_mean(tf.nn.l2_loss(y_p-tf.reshape(yq,[batch_size,mm])))
		   
		# learning rate decay
		global_step = tf.Variable(0, trainable=False)
		lr = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.95, staircase=True)
		reg = tf.nn.l2_loss(w_proj)
		opt_op = tf.train.AdamOptimizer(starter_learning_rate,beta1=beta1, beta2=beta2).minimize(loss, global_step=global_step)
		# Create an ExponentialMovingAverage object
		ema = tf.train.ExponentialMovingAverage(decay=0.9998)

		with tf.control_dependencies([opt_op]):
			training_op = ema.apply([w, w2, beta, w_proj])

		with tf.variable_scope('model', reuse=True):
			# use ema.average(var) to return the average of variable var
			wavg = ema.average(w)
			w2avg = ema.average(w2)
			betaavg = ema.average(beta)
			w_projavg = ema.average(w_proj)

			xd_in = tf.reshape(xd, (-1, 1, window, 1))
			direct_zx = tf.square(tf.nn.conv2d(xd_in,wsin,strides=[1,1,stride,1],padding='VALID')) \
				  + tf.square(tf.nn.conv2d(xd_in,wcos,strides=[1,1,stride,1],padding='VALID'))
			direct_zx = tf.nn.relu(tf.nn.conv2d(tf.log(direct_zx+10e-15),wavg,strides=[1,1,1,stride_y],padding='VALID',data_format='NCHW'))
			direct_zx = tf.nn.relu(tf.nn.conv2d(direct_zx,w2avg,strides=[1,1,1,1],padding='VALID',data_format='NCHW'))
			direct_zx = tf.matmul(tf.reshape(direct_zx,[tf.shape(xd_in)[0],int(num_regions3_x*num_regions3_y*k3)]),betaavg)
			direct_zx = tf.reshape(direct_zx, (-1, timesteps, mm))
			#apply lstm
			direct_outputs, direct_states  = tf.nn.bidirectional_dynamic_rnn(
			    cell_fw=cell,
			    cell_bw=cell,
			    dtype=tf.float32,
			    inputs=direct_zx)
			direct_output_fw, direct_output_bw = direct_outputs
			#compare only the outputs of middle of sequence
			direct_output_fw = direct_output_fw[:,int(timesteps/2),:]
			direct_output_bw = direct_output_bw[:,int(timesteps/2),:]
			y_pd = tf.matmul(tf.concat([direct_output_fw, direct_output_bw], -1),w_projavg)
			direct_loss = tf.reduce_mean(tf.nn.l2_loss(y_pd-yd))

		# train saver
		saver = tf.train.Saver()

	return graph, training_op, loss, global_step, lr, reg, y_pd, direct_loss, xb, yb, xd, yd, saver, w, wavg, w2, w2avg, beta, betaavg

def LSTM_PT_graph_AdamOpt_Reg(model_stats, batch_size, window, timesteps, stride, out, wsin, wcos, d2_x, d2_y, k2, stride_y, 
	d3_x, d3_y, k3, num_regions3_x, num_regions3_y, num_units, starter_learning_rate, decay_steps, lr_decay, beta1=0.9, beta2=0.999, beta_reg=0.01):
    
	mm = model_stats.mm
	graph = tf.Graph()

	with graph.as_default():
		# if the input is a batch
		with tf.variable_scope('data_queue'):
			xb = tf.placeholder(tf.float32, shape=[None, timesteps, 1, window+(out-1)*stride, 1])
			yb = tf.placeholder(tf.float32, shape=[None, timesteps, out, mm])
			# put batches and different timesteps in one dimension
			xq = tf.reshape(xb, (-1, 1, window+(out-1)*stride, 1))  
			# middle of sequence
			yq = yb[:,int(timesteps/2),:,:]
	
		# if the input is directly evaluated without training
		with tf.variable_scope('direct_data'):
			# batch and timestep dimension together for Convolution
			xd = tf.placeholder(tf.float32, shape=[None,timesteps,1,window,1]) 
			yd = tf.placeholder(tf.float32, shape=[None, mm])
	    
		print ('---- Weights ----')
		with tf.variable_scope('parameters'): #glorot  
			w = tf.get_variable('w', [d2_x,d2_y,1,k2], trainable=False) # 1 x 128 x 1 x 128 (16.384 values)
			print ('w',w)
			w2 = tf.get_variable('w2', [d3_x,d3_y,k2,k3], trainable=False) # 13 x 1 x 128 x 1024 (1.703.936 values)
			print ('w2',w2)
			beta = tf.get_variable('beta', [int(num_regions3_x*num_regions3_y*k3),mm], trainable=False) # (1*65*1024) x 128 (8.519.680 values)
			print ('beta',beta)
	
		with tf.variable_scope('parameters_train'): #glorot 
			w_proj = tf.get_variable('w_proj',[num_units*2, mm]) #256 x 128
			print ('w_proj',w_proj)
	
		model_stats.register_weights([w,w2,beta,w_proj],['w','w2','beta','w_proj'])

		cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units, use_peepholes=True, state_is_tuple=True)
		#TODO: num_proj=mm and num_units bigger

		print ('\n---- Layers ----')
		with tf.variable_scope('model'):
			##### Feed Forward Layers
			# data storage order: batch, height, width, channels
			zx = tf.square(tf.nn.conv2d(xq,wsin,strides=[1,1,stride,1],padding='VALID')) \
			   + tf.square(tf.nn.conv2d(xq,wcos,strides=[1,1,stride,1],padding='VALID'))
			print ('zx',zx)
			# data storage order: batch, channels, height, width
			zx = tf.nn.relu(tf.nn.conv2d(tf.log(zx+10e-15),w,strides=[1,1,1,stride_y],padding='VALID',data_format='NCHW'))
			print ('z2',zx)
			zx = tf.nn.relu(tf.nn.conv2d(zx,w2,strides=[1,1,1,1],padding='VALID',data_format='NCHW'))
			print ('z3',zx)
			zx = tf.matmul(tf.reshape(zx,[batch_size*timesteps,int(num_regions3_x*num_regions3_y*k3)]),beta)
			print ('y_feed_forward',zx)

			##### Long short-term memory unit: Dynamic Bidirectional Recurrent 
			# separate batch and time dimension again
			zx = tf.reshape(zx, (-1, timesteps, mm))
			print('\n---- LSTM-Cell ----')
			print('Number of units:',num_units)
			print('Shape of input:',zx.get_shape()) # [batch_size, max_time, ...]
			outputs, states  = tf.nn.bidirectional_dynamic_rnn(
			    cell_fw=cell,
			    cell_bw=cell,
			    dtype=tf.float32,
			    inputs=zx)
			output_fw, output_bw = outputs
			#compare only the outputs of middle of sequence
			output_fw = output_fw[:,int(timesteps/2),:]
			output_bw = output_bw[:,int(timesteps/2),:]
			print('Shape of output: 2 x',output_bw.get_shape()) # [batch_size, max_time, ...]

			##### Feed Forward Layer: Project to final output dimension
			# project forward and backward output to final output
			y_p = tf.matmul(tf.concat([output_fw, output_bw], -1),w_proj)
			print ('y_output:',y_p)
		
			##### Loss    
			loss = tf.reduce_mean(tf.nn.l2_loss(y_p-tf.reshape(yq,[batch_size,mm])))
		   
		# learning rate decay
		global_step = tf.Variable(0, trainable=False)
		lr = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.95, staircase=True)
		reg = tf.nn.l2_loss(w_proj)
		opt_op = tf.train.AdamOptimizer(starter_learning_rate,beta1=beta1, beta2=beta2).minimize(tf.reduce_mean(loss + beta_reg*reg), global_step=global_step)
		# Create an ExponentialMovingAverage object
		ema = tf.train.ExponentialMovingAverage(decay=0.9998)

		with tf.control_dependencies([opt_op]):
			training_op = ema.apply([w, w2, beta, w_proj])

		with tf.variable_scope('model', reuse=True):
			# use ema.average(var) to return the average of variable var
			wavg = ema.average(w)
			w2avg = ema.average(w2)
			betaavg = ema.average(beta)
			w_projavg = ema.average(w_proj)

			xd_in = tf.reshape(xd, (-1, 1, window, 1))
			direct_zx = tf.square(tf.nn.conv2d(xd_in,wsin,strides=[1,1,stride,1],padding='VALID')) \
				  + tf.square(tf.nn.conv2d(xd_in,wcos,strides=[1,1,stride,1],padding='VALID'))
			direct_zx = tf.nn.relu(tf.nn.conv2d(tf.log(direct_zx+10e-15),wavg,strides=[1,1,1,stride_y],padding='VALID',data_format='NCHW'))
			direct_zx = tf.nn.relu(tf.nn.conv2d(direct_zx,w2avg,strides=[1,1,1,1],padding='VALID',data_format='NCHW'))
			direct_zx = tf.matmul(tf.reshape(direct_zx,[tf.shape(xd_in)[0],int(num_regions3_x*num_regions3_y*k3)]),betaavg)
			direct_zx = tf.reshape(direct_zx, (-1, timesteps, mm))
			#apply lstm
			direct_outputs, direct_states  = tf.nn.bidirectional_dynamic_rnn(
			    cell_fw=cell,
			    cell_bw=cell,
			    dtype=tf.float32,
			    inputs=direct_zx)
			direct_output_fw, direct_output_bw = direct_outputs
			#compare only the outputs of middle of sequence
			direct_output_fw = direct_output_fw[:,int(timesteps/2),:]
			direct_output_bw = direct_output_bw[:,int(timesteps/2),:]
			y_pd = tf.matmul(tf.concat([direct_output_fw, direct_output_bw], -1),w_projavg)
			direct_loss = tf.reduce_mean(tf.nn.l2_loss(y_pd-yd))

		# train saver
		saver = tf.train.Saver()

	return graph, training_op, loss, global_step, lr, reg, y_pd, direct_loss, xb, yb, xd, yd, saver, w, wavg, w2, w2avg, beta, betaavg



# Graph using pretrained weights, sigmoid output functions and Adam Optimizer
def LSTM_PT_graph_AdamOpt_sig_noReg(model_stats, batch_size, window, timesteps, stride, out, wsin, wcos, d2_x, d2_y, k2, stride_y, 
	d3_x, d3_y, k3, num_regions3_x, num_regions3_y, num_units, starter_learning_rate, decay_steps, lr_decay, beta1=0.9, beta2=0.999, beta_reg=0):
    
	mm = model_stats.mm
	graph = tf.Graph()

	with graph.as_default():
		# if the input is a batch
		with tf.variable_scope('data_queue'):
			xb = tf.placeholder(tf.float32, shape=[None, timesteps, 1, window+(out-1)*stride, 1])
			yb = tf.placeholder(tf.float32, shape=[None, timesteps, out, mm])
			# put batches and different timesteps in one dimension
			xq = tf.reshape(xb, (-1, 1, window+(out-1)*stride, 1))  
			# middle of sequence
			yq = yb[:,int(timesteps/2),:,:]
	
		# if the input is directly evaluated without training
		with tf.variable_scope('direct_data'):
			# batch and timestep dimension together for Convolution
			xd = tf.placeholder(tf.float32, shape=[None,timesteps,1,window,1]) 
			yd = tf.placeholder(tf.float32, shape=[None, mm])
	    
		print ('---- Weights ----')
		with tf.variable_scope('parameters'): #glorot  
			w = tf.get_variable('w', [d2_x,d2_y,1,k2], trainable=False) # 1 x 128 x 1 x 128 (16.384 values)
			print ('w',w)
			w2 = tf.get_variable('w2', [d3_x,d3_y,k2,k3], trainable=False) # 13 x 1 x 128 x 1024 (1.703.936 values)
			print ('w2',w2)
			beta = tf.get_variable('beta', [int(num_regions3_x*num_regions3_y*k3),mm], trainable=False) # (1*65*1024) x 128 (8.519.680 values)
			print ('beta',beta)
	
		with tf.variable_scope('parameters_train'): #glorot 
			w_proj = tf.get_variable('w_proj',[num_units*2+1, mm]) #256+1 x 128
			print ('w_proj',w_proj)
	
		model_stats.register_weights([w,w2,beta,w_proj],['w','w2','beta','w_proj'])

		cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units, use_peepholes=True, state_is_tuple=True)
		#TODO: num_proj=mm and num_units bigger

		print ('\n---- Layers ----')
		with tf.variable_scope('model'):
			##### Feed Forward Layers
			# data storage order: batch, height, width, channels
			zx = tf.square(tf.nn.conv2d(xq,wsin,strides=[1,1,stride,1],padding='VALID')) \
			   + tf.square(tf.nn.conv2d(xq,wcos,strides=[1,1,stride,1],padding='VALID'))
			print ('zx',zx)
			# data storage order: batch, channels, height, width
			zx = tf.nn.relu(tf.nn.conv2d(tf.log(zx+10e-15),w,strides=[1,1,1,stride_y],padding='VALID',data_format='NCHW'))
			print ('z2',zx)
			zx = tf.nn.relu(tf.nn.conv2d(zx,w2,strides=[1,1,1,1],padding='VALID',data_format='NCHW'))
			print ('z3',zx)
			zx = tf.matmul(tf.reshape(zx,[batch_size*timesteps,int(num_regions3_x*num_regions3_y*k3)]),beta)
			print ('y_feed_forward',zx)

			##### Long short-term memory unit: Dynamic Bidirectional Recurrent 
			# separate batch and time dimension again
			zx = tf.reshape(zx, (-1, timesteps, mm))
			print('\n---- LSTM-Cell ----')
			print('Number of units:',num_units)
			print('Shape of input:',zx.get_shape()) # [batch_size, max_time, ...]
			outputs, states  = tf.nn.bidirectional_dynamic_rnn(
			    cell_fw=cell,
			    cell_bw=cell,
			    dtype=tf.float32,
			    inputs=zx)
			output_fw, output_bw = outputs
			#compare only the outputs of middle of sequence
			output_fw = output_fw[:,int(timesteps/2),:]
			output_bw = output_bw[:,int(timesteps/2),:]
			print('Shape of output: 2 x',output_bw.get_shape()) # [batch_size, max_time, ...]

			##### Feed Forward Layer: Project to final output dimension
			# project forward and backward output to final output and add BIAS
			y_p = tf.matmul(
				tf.concat([tf.concat([output_fw, output_bw], -1),tf.ones([batch_size,1], tf.float32)],-1)	
				,w_proj)
			print ('y_output:',y_p)

			##### Loss
			loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(yq,[batch_size,mm]), logits=y_p) 
		
		# learning rate decay (ONLY FAKE)
		global_step = tf.Variable(0, trainable=False)
		lr = tf.train.exponential_decay(starter_learning_rate, global_step, 1000000, 0, staircase=True)
		reg = tf.nn.l2_loss(w_proj)
		opt_op = tf.train.AdamOptimizer(starter_learning_rate,beta1=beta1, beta2=beta2).minimize(loss, global_step=global_step)

		# Create an ExponentialMovingAverage object
		ema = tf.train.ExponentialMovingAverage(decay=0.9998)

		with tf.control_dependencies([opt_op]):
			training_op = ema.apply([w, w2, beta, w_proj])

		with tf.variable_scope('model', reuse=True):
			# use ema.average(var) to return the average of variable var
			wavg = ema.average(w)
			w2avg = ema.average(w2)
			betaavg = ema.average(beta)
			w_projavg = ema.average(w_proj)

			xd_in = tf.reshape(xd, (-1, 1, window, 1))
			direct_zx = tf.square(tf.nn.conv2d(xd_in,wsin,strides=[1,1,stride,1],padding='VALID')) \
				  + tf.square(tf.nn.conv2d(xd_in,wcos,strides=[1,1,stride,1],padding='VALID'))
			direct_zx = tf.nn.relu(tf.nn.conv2d(tf.log(direct_zx+10e-15),wavg,strides=[1,1,1,stride_y],padding='VALID',data_format='NCHW'))
			direct_zx = tf.nn.relu(tf.nn.conv2d(direct_zx,w2avg,strides=[1,1,1,1],padding='VALID',data_format='NCHW'))
			direct_zx = tf.matmul(tf.reshape(direct_zx,[tf.shape(xd_in)[0],int(num_regions3_x*num_regions3_y*k3)]),betaavg)
			direct_zx = tf.reshape(direct_zx, (-1, timesteps, mm))
			#apply lstm
			direct_outputs, direct_states  = tf.nn.bidirectional_dynamic_rnn(
			    cell_fw=cell,
			    cell_bw=cell,
			    dtype=tf.float32,
			    inputs=direct_zx)
			direct_output_fw, direct_output_bw = direct_outputs
			#compare only the outputs of middle of sequence
			direct_output_fw = direct_output_fw[:,int(timesteps/2),:]
			direct_output_bw = direct_output_bw[:,int(timesteps/2),:]
            
			y_pd = tf.sigmoid(tf.matmul(
				tf.concat([tf.concat([direct_output_fw, direct_output_bw], -1),tf.ones([tf.shape(xd)[0],1], tf.float32)],1)
				,w_projavg))
			direct_loss = tf.reduce_mean(tf.nn.l2_loss(y_pd-yd))

		# train saver
		saver = tf.train.Saver()

	return graph, training_op, loss, global_step, lr, reg, y_pd, direct_loss, xb, yb, xd, yd, saver, w, wavg, w2, w2avg, beta, betaavg

##########################################
# functions to handle result-diccionary

# in case diccionary should be printed
def print_dict(dictionary):
    for key, value in  dictionary.items():
        print (key)
        print (value)
#print_dict(model_stats.stats)
    
#save diccionary
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# load diccionary
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

    
def mirex_statistics(Y, Y_p, threshold=.5, base_note=0, mm=128):
    
    avp = average_precision_score(Y.flatten(), Y_p.flatten())

    # we want zeros and ones
    Yhat_p = Y_p>threshold
    
    # obtain frequencies
    Yp_list = []
    Y_list = []
    
    # for every sample
    for i in range(len(Yhat_p)):
        fhat = []
        f = []
        
        for note in range(mm):
            if Yhat_p[i][note] == 1:
                fhat.append(440.*2**(((note+base_note) - 69.)/12.))
            if Y[i][note] == 1:
                f.append(440.*2**(((note+base_note) - 69.)/12.))

        Yp_list.append(np.array(fhat))
        Y_list.append(np.array(f))

    # Precision, Recall, Accuracy, Substitution, Miss, False Alarm, and Total Error scores 
    # based both on raw frequency values and values mapped to a single octave (chroma)
    
    P,R,Acc,Esub,Emiss,Efa,Etot,cP,cR,cAcc,cEsub,cEmiss,cEfa,cEtot = \
    mir_eval.multipitch.metrics(np.arange(len(Y_list))/100., Y_list,
                                np.arange(len(Yp_list))/100., Yp_list)


    print('{:.2f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
        threshold,100*avp,100*P,100*R,Acc,Etot,Esub,Emiss,Efa))

    return avp,P,R,Acc,Etot

################################################################
# runtime and learning rate

def plot_lr(results_stats, plot_characteristics, vers):

    iters_res = results_stats['iter'][2]
    time_res = results_stats['time'][2]
    lr_res = results_stats['lr'][2]

    plt.figure(figsize=(plot_characteristics['width'], plot_characteristics['height']/2))

    plt.subplot(1, 1, 1)
    plt.plot(iters_res[1:], lr_res[1:], c="darkblue")
    plt.ylabel('Learning rate',fontsize=plot_characteristics['fontsize'])
    plt.xlabel('Iteration',fontsize=plot_characteristics['fontsize'])
    if plot_characteristics['savefig']:
        plt.savefig('results/plots/'+vers+'_runtime_lr.png')
        
def plot_lr_runtime(results_stats, plot_characteristics, vers):

    iters_res = results_stats['iter'][2]
    time_res = results_stats['time'][2]
    lr_res = results_stats['lr'][2]

    plt.figure(figsize=(plot_characteristics['width'], plot_characteristics['height']))

    #plot time needed for 'control_step' steps
    plt.subplot(2, 1, 1)
    plt.title('Runtime and Learning Rate', fontsize=plot_characteristics['fontsize'])
    plt.plot(iters_res[1:], time_res[1:], c="darkblue")
    plt.ylabel('Runtime of last '+str(plot_characteristics['control_step'])+' steps (sec)')
    plt.ylim( (0.9*np.min(time_res[1:]), 1.1*np.max(time_res[1:])))

    plt.subplot(2, 1, 2)
    plt.plot(iters_res[1:], lr_res[1:], c="darkblue")
    plt.ylabel('Learning rate',fontsize=plot_characteristics['fontsize'])
    plt.xlabel('Iteration',fontsize=plot_characteristics['fontsize'])
    
    if plot_characteristics['savefig']:
        plt.savefig('results/plots/'+vers+'_runtime_lr.png')


################################################################
# error and precision

def plot_err_prec(results_stats, plot_characteristics, vers):
    fontsize = plot_characteristics['fontsize']
    
    iters_res = results_stats['iter'][2]
    mse_train_res = results_stats['mse_train'][2]
    mse_train_batch = results_stats['mse_train_1'][2]
    mse_test_res = results_stats['mse_test'][2]
    avp_train_res = results_stats['avp_train'][2]
    avp_test_res = results_stats['avp_test'][2]

    plt.figure(figsize=(plot_characteristics['width'], plot_characteristics['height']))
    
    #plot mean squared error
    plt.subplot(2, 1, 1)
    #plt.title('Error and Precision',fontsize=fontsize)
    plt.plot(iters_res, mse_train_res, label='Training', c="c")
    if np.shape(results_stats['mse_train_1'][2])[-1] != 128:
        plt.plot(iters_res, mse_train_batch, label='Batch', c="b")
    plt.plot(iters_res, mse_test_res, label='Test', c="darkblue")
    plt.legend(fontsize=fontsize)
    plt.ylabel('Mean Square Error',fontsize=fontsize)

    #plot average precision
    plt.subplot(2, 1, 2)
    plt.plot(iters_res, avp_train_res, label='Training', c="c")
    plt.plot(iters_res, avp_test_res, label='Test', c="darkblue")
    plt.legend(fontsize=fontsize)
    plt.ylabel('Average Precision',fontsize=fontsize)
    plt.xlabel('Iteration',fontsize=fontsize)
    if plot_characteristics['savefig']:
        plt.savefig('results/plots/'+vers+'_error_prec.png')

def plot_prec(results_stats, plot_characteristics, vers):
    fontsize = plot_characteristics['fontsize']
    
    iters_res = results_stats['iter'][2]
    avp_train_res = results_stats['avp_train'][2]
    avp_test_res = results_stats['avp_test'][2]

    plt.figure(figsize=(plot_characteristics['width'], plot_characteristics['height']/2))
    
    #plot mean squared error
    plt.subplot(1, 1, 1)
    #plt.title('Error and Precision',fontsize=fontsize)
    plt.plot(iters_res, avp_train_res, label='Training', c="c")
    plt.plot(iters_res, avp_test_res, label='Test', c="darkblue")
    plt.legend(fontsize=fontsize)
    plt.ylabel('Average Precision',fontsize=fontsize)
    plt.xlabel('Iteration',fontsize=fontsize)
    if plot_characteristics['savefig']:
        plt.savefig('results/plots/'+vers+'_error_prec.png')

################################################################
# norms of the weights

def plot_norms_lstm(results_stats, plot_characteristics, vers): # plot the additional output layer
    fontsize = plot_characteristics['fontsize']
    iters_res = results_stats['iter'][2]
    nw = results_stats['nw'][2]
    nw2 = results_stats['nw2'][2]
    nbeta = results_stats['nbeta'][2]
    nw_proj = results_stats['nw_proj'][2]

    plt.figure(figsize=(plot_characteristics['width'], plot_characteristics['height']))
    
    plt.subplot(3, 1, 1)
    plt.plot(iters_res[1:], nw[1:], label='2nd layer', c="darkblue")
    plt.plot(iters_res[1:], nw2[1:], label='3rd layer', c="c")
    plt.legend(fontsize=fontsize)
    plt.ylabel('l2-Norm',fontsize=fontsize)

    plt.subplot(3, 1, 2)
    plt.plot(iters_res[1:], nbeta[1:], label='4th layer',c="darkblue")
    plt.ylabel('l2-Norm',fontsize=fontsize)
    plt.legend(fontsize=fontsize)

    plt.subplot(3, 1, 3)
    plt.plot(iters_res[1:], nw_proj[1:], label='Output layer', c="c")
    plt.legend(fontsize=fontsize)
    plt.ylabel('l2-Norm',fontsize=fontsize)
    plt.xlabel('Iteration',fontsize=fontsize)
    if plot_characteristics['savefig']:
        plt.savefig('results/plots/'+vers+'_weight_norms.png')

def plot_norms(results_stats, plot_characteristics, vers): # plot the additional output layer
    fontsize = plot_characteristics['fontsize']
    iters_res = results_stats['iter'][2]
    nw = results_stats['nw'][2]
    nw2 = results_stats['nw2'][2]
    nbeta = results_stats['nbeta'][2]

    plt.figure(figsize=(plot_characteristics['width'], plot_characteristics['height']))
    
    plt.subplot(2, 1, 1)
    plt.plot(iters_res[1:], nw[1:], label='2nd layer', c="darkblue")
    plt.plot(iters_res[1:], nw2[1:], label='3rd layer', c="c")
    plt.legend(fontsize=fontsize)
    plt.ylabel('l2-Norm',fontsize=fontsize)

    plt.subplot(2, 1, 2)
    plt.plot(iters_res[1:], nbeta[1:], label='output layer',c="darkblue")
    plt.ylabel('l2-Norm',fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.xlabel('Iteration',fontsize=fontsize)
    if plot_characteristics['savefig']:
        plt.savefig('results/plots/'+vers+'_weight_norms.png')

def plot_out_norms(results_stats, plot_characteristics, vers): # plot the additional output layer
    fontsize = plot_characteristics['fontsize']
    iters_res = results_stats['iter'][2]
    nw_proj = results_stats['nw_proj'][2]

    plt.figure(figsize=(plot_characteristics['width'], plot_characteristics['height']/3))
    plt.plot(iters_res[1:], nw_proj[1:], label='Output layer', c="c")
    plt.legend(fontsize=fontsize)
    plt.ylabel('l2-Norm',fontsize=fontsize)
    plt.xlabel('Iteration',fontsize=fontsize)
    if plot_characteristics['savefig']:
        plt.savefig('results/plots/'+vers+'_weight_norms.png')


##################################################################
# plot the scores

def scores(y_sample, sample_test_id, start = 1, sec=10, fs=11250, stride=512, downsample=True,cmap='terrain_r',savefig=False):
    times = []
    wps = fs / float(stride)
    if downsample==True:
        Yvec = np.zeros((int(wps*sec),128))
        for ww in range(Yvec.shape[0]):
            times += [ww*stride+start*fs]
            ll = y_sample[ww*stride+start*fs]
            for label in ll:
                Yvec[ww,label.data[1]] = 1
    else:
        Yvec=y_sample
    fig = plt.figure(figsize=(20,10))
    plt.imshow(Yvec.T,aspect='auto',cmap=cmap)
    plt.gca().invert_yaxis()
    new_tick_locations = np.linspace(0,1,11)
    fig.axes[0].set_xticks(new_tick_locations*sec*wps)
    fig.axes[0].set_xticklabels((new_tick_locations*10+start).astype(int))
    fig.axes[0].set_xlabel('Time in seconds',fontsize=24)
    fig.axes[0].set_ylabel('Note (MIDI code)',fontsize=24)
    #plt.title("Recording "+str(sample_test_id),fontsize=24)
    if savefig != False:
        plt.savefig(savefig)
    return times, Yvec

def compare_scores(y, y_p, sample_test_id, start = 1, sec=10, fs=11250, stride=512,cmap='ocean_r',savefig=False):
    if np.shape(y)!=np.shape(y_p):
        print('input vectors have to bet of the same shape')
        return 1
    wps = fs / float(stride)
    Yvec = np.copy(y)
    count_3 = 0
    count_2 = 0
    for ww in range(Yvec.shape[0]):
        for ll in range(Yvec.shape[1]):
            if y[ww,ll] == 1:
                if y_p[ww,ll] != 1:
                    Yvec[ww,ll] = 0.4
            else:
                if y_p[ww,ll] == 1:
                    Yvec[ww,ll] = 0.2
    fig = plt.figure(figsize=(20,10))
    plt.imshow(Yvec.T,aspect='auto',cmap=cmap)
    plt.gca().invert_yaxis()
    new_tick_locations = np.linspace(0,1,11)
    fig.axes[0].set_xticks(new_tick_locations*sec*wps)
    fig.axes[0].set_xticklabels((new_tick_locations*10+start).astype(int))
    fig.axes[0].set_xlabel('Time in seconds',fontsize=24)
    fig.axes[0].set_ylabel('Note (MIDI code)',fontsize=24)
    plt.title("Black: Corect, Yellow: False Alarm, Orange: False Negative",fontsize=24)
    if savefig != False:
        plt.savefig(savefig)
    return Yvec
