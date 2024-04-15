from scipy.sparse import csr_matrix
import tarfile

import torch
import numpy as np
import scipy.sparse as sp
import itertools
import math
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from models import *
from scipy import sparse
import random
from os import listdir
from os.path import isfile, join
import sklearn.metrics as metrics
import copy
from sklearn.metrics import precision_recall_curve
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import pickle
import time
import warnings
import argparse
import yaml
import os
import logging
warnings.filterwarnings('ignore')
import pandas as pd
import pickle
import networkx

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-f", "--config_file", default="../configs/config-1.yaml", help="Configuration file to load.")
ARGS = parser.parse_args()

with open(ARGS.config_file, 'r') as f:
      config = yaml.safe_load(f)
print("Loaded configuration file", ARGS.config_file)



name = 'Results/'+ARGS.config_file.split('/')[1].split('.')[0]+'_Slashdot'
time_list = config['time_list']
L_list = config['L_list']
K = config['K']
p_val = config['p_val']
p_nodes = config['p_nodes']
p_test = config['p_test']
n_hidden = config['n_hidden']
max_iter = config['max_iter']
# tolerance_init = config['tolerance']
tolerance_init = 100
save_time_complex = config['save_time_complex']
save_MRR_MAP = config['save_MRR_MAP']
save_sigma_mu = config['save_sigma_mu']
scale=False
seed=0
verbose=True


def init_logging_handler(exp_name):
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('{}/log_{}.txt'.format(exp_name, current_time))
    logging.basicConfig(handlers=[stderr_handler, file_handler])
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

init_logging_handler(name)
logging.debug(str(config))

def check_if_gpu():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = check_if_gpu()
logging.debug('The code will be running on {}'.format(device))




def is_compatible(filename):
    return any(filename.endswith(extension) for extension in ['.txt'])


class dataset_mit(torch.utils.data.Dataset):
    def __init__(self, root_dir, train = True):
      
        self.root_dir = root_dir

        self.Adj_arr = []
        self.X_Sparse_arr = []
        count = 0
        max_size = 0
        


        with open(self.root_dir + 'datasets/slashdot_monthly_dynamic.pkl', 'rb') as f:
            data = pickle.load(f)
        
        c = 0
        source_arr = []
        time_arr = []
        target_arr = []
        for network in data:
            for i in data[network].edges:
                time_arr.append(c)
                source_arr.append(i[0])
                target_arr.append(i[1])
            c = c+1
            
        df = pd.DataFrame({'Source': source_arr, 'Target': target_arr,'time':time_arr})
        df.columns = ['source', 'target','time']
        
        for i in range(df['time'].max()+1):
            logging.debug(count)
            df1 = df[df['time']== i]
            arr = df1[['source', 'target']].to_numpy()
            
            A, X_Sparse, size = self.get_graph(arr, max_size)
            if size > max_size:
                max_size = size
                
            self.Adj_arr.append(A)
            self.X_Sparse_arr.append(X_Sparse)
            count = count + 1
            
            
    def __len__(self):
        return len(self.img_filename)
    
    
    def __getitem__(self, idx):
       
        
        return self.Adj_arr[idx], self.X_Sparse_arr[idx]
    
    def get_graph(self,arr, max_size):

        new_a = arr
        if max_size > arr.max()+1:
            new_max = max_size
            arr_zero = np.zeros((max_size, max_size))
        else:
            arr_zero = np.zeros((arr.max()+1, arr.max()+1))
            new_max =  arr.max()+1
        for i in new_a:
            arr_zero[int(i[0])][int(i[1])] = 1 
        arr_zero[range(len(arr_zero)), range(len(arr_zero))] = 0
        A = csr_matrix(arr_zero)
        X= A + sp.eye(A.shape[0])
        X_Sparse = sparse_feeder(X)
        X_Sparse = spy_sparse2torch_sparse(X_Sparse)
        
        return A, X_Sparse, new_max


# In[4]:


data = dataset_mit('../')


def sample_zero_forever(mat):
    nonzero_or_sampled = set(zip(*mat.nonzero()))
    while True:
        t = tuple(np.random.randint(0, mat.shape[0], 2))
        if t not in nonzero_or_sampled:
            yield t
            nonzero_or_sampled.add(t)

def sample_zero_n(mat, n=1000):
    itr = sample_zero_forever(mat)
    return [next(itr) for _ in range(n)]




def get_row_MRR(probs,true_classes):
    existing_mask = true_classes == 1
        #descending in probability
    ordered_indices = np.flip(probs.argsort())

    ordered_existing_mask = existing_mask[ordered_indices]

    existing_ranks = np.arange(1,
                                   true_classes.shape[0]+1,
                                   dtype=np.float)[ordered_existing_mask]

    MRR = (1/existing_ranks).sum()/existing_ranks.shape[0]
    return MRR

def get_MRR(predictions,true_classes, adj):
    probs = predictions

    probs = probs.detach().cpu().numpy()
    true_classes = true_classes
    adj = adj


    pred_matrix = coo_matrix((probs,(adj[0],adj[1]))).toarray()
    true_matrix = coo_matrix((true_classes,(adj[0],adj[1]))).toarray()

    row_MRRs = []
    for i, pred_row in enumerate(pred_matrix):
            #check if there are any existing edges
        if np.isin(1,true_matrix[i]):
            row_MRRs.append(get_row_MRR(pred_row,true_matrix[i]))

    avg_MRR = torch.tensor(row_MRRs).mean()
    return avg_MRR

def get_MAP_e(predictions,true_classes, adj):

    probs = predictions
    probs = probs.detach().cpu().numpy()
    true_classes = true_classes
    adj = adj

    var = average_precision_score(true_classes, probs)

    return var



if save_time_complex == True:
    as_time = []


if save_sigma_mu == True:
    sigma_L_arr = []
    mu_L_arr = []

for L in L_list:
    if save_time_complex == True:
        time_timestamp = []
    count = 0
    sigma_timestamp = []
    mu_timestamp = []
    
    for i in data:
        logging.debug('timestamp {}'.format(count))
        A = i[0]
        X_Sparse = i[1]
        val_ones = list(set(zip(*A.nonzero())))
        val_ones = random.sample(val_ones, A.nnz)
        val_ones = [list(ele) for ele in val_ones]
        val_zeros = sample_zero_n(A,A.nnz)
        val_zeros = [list(ele) for ele in val_zeros]

        N, D = A.shape

        hops = get_hops(A, K)

        scale_terms = {h if h != -1 else max(hops.keys()) + 1:
                   hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                           for h in hops}
        learning_rate = 1e-3
        start = time.time()
        if count == 0:
            net = Graph2Gauss_Torch(n_hidden, L, D)
        else:
            net = copy.deepcopy(net)
        if net.net[0].weight.data.shape[0] < D:
            dummy_input = InputLinear(net.net[0].weight.data.shape[0])
            dummy_output, net.net[0] = wider(dummy_input, net.net[0], D)
        net = net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        early_stopping_score_max = -float('inf')
        count = count + 1
        tolerance = tolerance_init
        
#         if count <89 and count >2:
        if count <=8:
            logging.debug('Training')
#             net.train()

            for epoch in range(max_iter):
                optimizer.zero_grad()
                X_Sparse = X_Sparse.to(device)
                encoded, mu, sigma = net(X_Sparse)
                triplets_new,scale_terms_new = to_triplets(sample_all_hops(hops), scale_terms)
                loss_s =  build_loss(triplets_new, scale_terms_new, mu, sigma, L, scale = scale)
                if loss_s > 1000:
                    logging.debug('Resetting')
                    net = Graph2Gauss_Torch(n_hidden, L, D).to(device)
                    encoded, mu, sigma = net(X_Sparse)
                    triplets_new,scale_terms_new = to_triplets(sample_all_hops(hops), scale_terms)
                    loss_s =  build_loss(triplets_new, scale_terms_new, mu, sigma, L, scale = scale)
                if p_val > 0:
                    val_edges = np.row_stack((val_ones, val_zeros))
                    try:
                        neg_val_energy = -Energy_KL(mu, sigma, val_edges, L)
                        val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1
                    
                    except:
                        print("Some Error")
                    val_early_stopping = True
                else:
                    val_early_stopping = False

                        # setup the test set for easy evaluation
                if p_test > 0:
                    test_edges = np.row_stack((test_ones, test_zeros))
                    neg_test_energy = -Energy_KL(mu, sigma,test_edges, L)
                    test_ground_truth = A[test_edges[:, 0], test_edges[:, 1]].A1
                    val_early_stopping = True

                if val_early_stopping:
                    try:
                        val_auc, val_ap = score_link_prediction(val_ground_truth, neg_val_energy.cpu().detach().numpy())
                        early_stopping_score = val_auc + val_ap
                        
                    except:
                        print("cannot show")
                    early_stopping_score = 0

                    if verbose and epoch % 1 == 0:
                        try:
                            logging.debug('Time: {}, L: {}, epoch: {:3d}, loss: {:.4f}, val_auc: {:.4f}, val_ap: {:.4f}'.format(count,L,epoch, loss_s.item(), val_auc, val_ap))
                        except:
                            print("cannot show")
                            print('Time',count)

                else:
                    early_stopping_score = -loss
                    if verbose and epoch % 1 == 0:
                        logging.debug('epoch: {:3d}, loss: {:.4f}'.format(epoch, loss))

                if early_stopping_score > early_stopping_score_max:
                    early_stopping_score_max = early_stopping_score
                    tolerance = tolerance
                else:
                    tolerance -= 1

                if tolerance == 0:
                    break


                loss_s.backward()
                optimizer.step()
        else:
            logging.debug('Testing')
            X_Sparse = X_Sparse.to(device)
            encoded, mu, sigma = net(X_Sparse)
            
        end = time.time()
        diff = end - start
        time_timestamp.append(diff)
        
    
        sigma_timestamp.append(sigma.cpu().detach().numpy())
        mu_timestamp.append(mu.cpu().detach().numpy())
        if p_val > 0:
                val_edges = np.row_stack((val_ones, val_zeros))
                try:
                    neg_val_energy = -Energy_KL(mu, sigma, val_edges, L)
                    val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1
                    val_early_stopping = True
                except:
                    logging.debug('Some error')
        else:
                val_early_stopping = False

        if p_test > 0:
                test_edges = np.row_stack((test_ones, test_zeros))
                neg_test_energy = -Energy_KL(mu, sigma,test_edges, L)
                test_ground_truth = A[test_edges[:, 0], test_edges[:, 1]].A1

        try:
            val_auc, val_ap = score_link_prediction(val_ground_truth, neg_val_energy.cpu().detach().numpy())
            logging.debug('val_auc {:.4f}, val_ap: {:.4f}'.format(val_auc, val_ap))
        except:
            logging.debug('Some error')

    if save_time_complex:
        as_time.append(time_timestamp)
    if save_sigma_mu == True:
        sigma_L_arr.append(sigma_timestamp)
        mu_L_arr.append(mu_timestamp)


if save_sigma_mu == True:
    if not os.path.exists(name+'/Eval_Results/saved_array'):
        os.makedirs(name+'/Eval_Results/saved_array')
    with open(name+'/Eval_Results/saved_array/sigma_as','wb') as f: pickle.dump(sigma_L_arr, f)
    with open(name+'/Eval_Results/saved_array/mu_as','wb') as f: pickle.dump(mu_L_arr, f)

if save_time_complex:
    if not os.path.exists(name+'/Eval_Results/saved_array'):
        os.makedirs(name+'/Eval_Results/saved_array')
    with open(name+'/Eval_Results/saved_array/as_time','wb') as f: pickle.dump(as_time, f)
        
        
        
