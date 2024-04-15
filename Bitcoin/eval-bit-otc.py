#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.sparse import csr_matrix
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
from copy import deepcopy
import json




#Read parameters from json file
f = open("config.json")
config = json.load(f)

K                 = config["K"]
p_val             = config["p_val"]
p_nodes           = config["p_nodes"]
n_hidden          = config["n_hidden"]
max_iter          = config["max_iter"]
tolerance_init    = config["tolerance"]
time_list         = config["time_list"]
L_list            = config["L_list"]
save_time_complex = config["save_time_complex"]
save_MRR_MAP      = config["save_MRR_MAP"]
save_sigma_mu     = config["save_sigma_mu"]
scale             = config["scale"]
seed              = config["seed"]
verbose           = config["verbose"]

lookback = config["lookback"]


# In[11]:


def init_logging_handler(exp_name):
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('{}/log_{}.txt'.format(exp_name, current_time))
    logging.basicConfig(handlers=[stderr_handler, file_handler])
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

name = 'Results/Bitcoin'
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


class Namespace(object):
    '''
    helps referencing object in a dictionary as dict.key instead of dict['key']
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)
        
        
def aggregate_by_time(time_vector,time_win_aggr):
    time_vector = time_vector - time_vector.min()
    time_vector = time_vector // time_win_aggr
    return time_vector


def cluster_negs_and_positives(ratings):
    pos_indices = ratings > 0
    neg_indices = ratings <= 0
    ratings[pos_indices] = 1
    ratings[neg_indices] = -1
    return ratings


def is_compatible(filename):
    return any(filename.endswith(extension) for extension in ['.txt'])


class dataset_bit_alpha(torch.utils.data.Dataset):
    def __init__(self, root_dir, train = True):
      
        self.root_dir = root_dir

        self.Adj_arr = []
        self.X_Sparse_arr = []
        count = 0
        max_size = 0
        file = self.root_dir+'/datasets/soc-sign-bitcoinotc.csv'
        with open(file) as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines]
        edges = torch.tensor(edges,dtype = torch.long)
        
        ecols = Namespace({'FromNodeId': 0,
                                  'ToNodeId': 1,
                                  'Weight': 2,
                                  'TimeStep': 3
                                })
        
        new_edges = edges[:,[ecols.FromNodeId,ecols.ToNodeId]]
        _, new_edges = new_edges.unique(return_inverse=True)
        edges[:,[ecols.FromNodeId,ecols.ToNodeId]] = new_edges
        
        
        timesteps = aggregate_by_time(edges[:,ecols.TimeStep],1200000)
        
        max_time = timesteps.max()
        min_time = timesteps.min()
        edges[:,ecols.TimeStep] = timesteps
        
        
        edges[:,ecols.Weight] = cluster_negs_and_positives(edges[:,ecols.Weight])
        
        
        #add the reversed link to make the graph undirected
#         edges = torch.cat([edges,edges[:,[ecols.ToNodeId,
#                                           ecols.FromNodeId,
#                                           ecols.Weight,
#                                           ecols.TimeStep]]])
        
        
        df = pd.DataFrame(edges.numpy())
        df.columns = ['source', 'target','weight','time']
        
        for i in range(df['time'].max()+1):
            print(count)
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



# In[14]:


data = dataset_bit_alpha('..')

# In[3]:


def sample_zero_forever(mat):
    nonzero_or_sampled = set(zip(*mat.nonzero()))
    while True:
        t = tuple(np.random.randint(0, mat.shape[0], 2))
        if t not in nonzero_or_sampled:
            yield t
            nonzero_or_sampled.add(t)

def sample_zero_n(mat, n=2000):
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
    probs = torch.sigmoid(predictions)

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
    
    probs = torch.sigmoid(predictions)
    probs = probs.detach().cpu().numpy()
    true_classes = true_classes
    adj = adj
    
    var = average_precision_score(true_classes, probs)

    return var
    


# In[4]:


import pickle


# In[5]:


# with open('config-1/Eval_Results/saved_array/mu_as','rb') as f: mu_64 = pickle.load(f)
# with open('config-1/Eval_Results/saved_array/sigma_as','rb') as f: sigma_64 = pickle.load(f)


# # In[6]:


# mu_64 = mu_64[2]
# sigma_64 = sigma_64[2]
# with open('test_bit_otc_all_config-1/Eval_Results/saved_array/mu_as','rb') as f: mu_arr = pickle.load(f)
# with open('test_bit_otc_all_config-1/Eval_Results/saved_array/sigma_as','rb') as f: sigma_arr = pickle.load(f)

name_loaded = 'Results/Bitcoin'
with open(name_loaded+'/Eval_Results/saved_array/mu_as','rb') as f: mu_arr = pickle.load(f)
with open(name_loaded+'/Eval_Results/saved_array/sigma_as','rb') as f: sigma_arr = pickle.load(f)
    

MAP_l = []
MRR_l = []
# time_list = [109, 112, 115, 118, 121, 124,129,132,135]
for l_num in range(len(L_list)):
    
    mu_64 = mu_arr[l_num]
    sigma_64 = sigma_arr[l_num]

    def unison_shuffled_copies(a, b, seed):
        assert len(a) == len(b)
        np.random.seed(seed)
        p = np.random.permutation(len(a))
        return a[p], b[p]




    class Classifier(torch.nn.Module):
        def __init__(self):
            super(Classifier,self).__init__()
            activation = torch.nn.ReLU()

            self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features = np.array(mu_64[0]).shape[1]*2,
                                                           out_features = np.array(mu_64[0]).shape[1]),
                                           activation,
                                           torch.nn.Linear(in_features = np.array(mu_64[0]).shape[1],
                                                           out_features = 1))

        def forward(self,x):
            return self.mlp(x)


    # In[14]:

    seed = 5
    torch.cuda.manual_seed_all(seed)
    classify = Classifier()
    classify.to(device)

    # loss = torch.nn.BCEWithLogitsLoss()
    loss = torch.nn.BCEWithLogitsLoss(reduce=False)

    optim = torch.optim.Adam(classify.parameters(), lr = 1e-3)
    mult = 10
    mult_test = 50
    num_epochs = 50
    for epoch in range (num_epochs):
    #     for i in range (1, len(val_timestep) - 30):
            count = 0
            for ctr in range(lookback+1,95):
                A_node = data[ctr][0].shape[0]
                A = data[ctr][0]

                if count > 0:
                    if A_node > A_prev_node:
                        A = A[:A_prev_node,:A_prev_node]

    #                 if ctr < 95 and ctr>62:
                    if ctr < 95:
                        logging.debug('Training')
                        logging.debug(ctr)

                        ones_edj = A.nnz
                        if A.shape[0]*mult<=(A.shape[0]-1)*(A.shape[0]-1):
                            zeroes_edj = A.shape[0]*mult
                        else:
                            zeroes_edj = (A.shape[0]-1)*(A.shape[0]-1) - A.nnz

                        tot = ones_edj + zeroes_edj


                        val_ones = list(set(zip(*A.nonzero())))
                        val_ones = random.sample(val_ones, ones_edj)
                        val_ones = [list(ele) for ele in val_ones] 
                        val_zeros = sample_zero_n(A,zeroes_edj)
                        val_zeros = [list(ele) for ele in val_zeros] 
                        val_edges = np.row_stack((val_ones, val_zeros))

                        val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1

                        a, b = unison_shuffled_copies(val_edges,val_ground_truth, count)     

                        if ctr > 0:
                            a_embed = np.array(mu_64[ctr-(lookback+1)])[a.astype(int)]
                            a_embed_sig = np.array(sigma_64[ctr-(lookback+1)])[a.astype(int)]


                            classify.train()

                            inp_clf = []
                            for d_id in range (tot):
                                inp_clf.append(np.concatenate((a_embed[d_id][0], a_embed[d_id][1]), axis = 0))

                            inp_clf = torch.tensor(np.asarray(inp_clf))

                            inp_clf = inp_clf.to(device)
                            out = classify(inp_clf).squeeze()

                            weight = torch.tensor([0.05, 0.95]).to(device)


                            label = torch.tensor(np.asarray(b)).to(device)

                            weight_ = weight[label.data.view(-1).long()].view_as(label)


                            l = loss(out, label)


                            l = l  * weight_
                            l = l.mean()

                            optim.zero_grad()

                            l.backward()
                            optim.step()
    #                         MRR = get_MAP_e(out.cpu(), label.cpu(), val_edges)

    #                         logging.debug('Epoch: {}, Timestep: {}, Loss: {}, MAP: {}, MRR: {}'.format(epoch, ctr, l.item(), get_MAP_e(out.cpu(), label.cpu(), None),MRR))

                            MRR = get_MRR(out.cpu(), label.cpu(), np.transpose(a))


                            logging.debug('L:{}, Epoch: {}, Timestep: {}, Loss: {}, MAP: {}, MRR: {}'.format(np.array(mu_64[0]).shape[1],epoch, ctr, l.item(), get_MAP_e(out.cpu(), label.cpu(), None),MRR))


                A_prev_node = data[ctr][0].shape[0]
                count = count+1

    # In[ ]:


    num_epochs = 1
    MAP_time = []
    MRR_time = []
    time_ctr = 0
    for epoch in range (num_epochs):
        get_MAP_avg = []
        get_MRR_avg = []
    #     for i in range (70, len(val_timestep)):
        count = 0
        for ctr in range(109,137):

            A_node = data[ctr][0].shape[0]
            A = data[ctr][0]

            if count > 0:
                if A_node > A_prev_node:
                    A = A[:A_prev_node,:A_prev_node]

                if ctr >= 109:
                    logging.debug('Testing')
                    logging.debug(ctr)

                    ones_edj = A.nnz
                    if A.shape[0]*mult_test<=(A.shape[0]-1)*(A.shape[0]-1):
                        zeroes_edj = A.shape[0]*mult_test
                    else:
                        zeroes_edj = (A.shape[0]-1)*(A.shape[0]-1) - A.nnz

                    tot = ones_edj + zeroes_edj

                    val_ones = list(set(zip(*A.nonzero())))
                    val_ones = random.sample(val_ones, ones_edj)
                    val_ones = [list(ele) for ele in val_ones] 
                    val_zeros = sample_zero_n(A,zeroes_edj)
                    val_zeros = [list(ele) for ele in val_zeros] 
                    val_edges = np.row_stack((val_ones, val_zeros))

                    val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1

                    a, b = unison_shuffled_copies(val_edges,val_ground_truth, count)   

                    if ctr > 0:

                        a_embed = np.array(mu_64[ctr-(lookback+1)])[a.astype(int)]
                        a_embed_sig = np.array(sigma_64[ctr-(lookback+1)])[a.astype(int)]

                        classify.eval()

                        inp_clf = []
                        for d_id in range (tot):
                            inp_clf.append(np.concatenate((a_embed[d_id][0], a_embed[d_id][1]), axis = 0))

                        inp_clf = torch.tensor(np.asarray(inp_clf))

                        inp_clf = inp_clf.to(device)
                        with torch.no_grad():
                            out = classify(inp_clf).squeeze()


                        weight = torch.tensor([0.05, 0.95]).to(device)


                        label = torch.tensor(np.asarray(b)).to(device)

                        weight_ = weight[label.data.view(-1).long()].view_as(label)



                        l = loss(out, label)

                        l = l  * weight_
                        l = l.mean()

                        
                        MAP_val = get_MAP_e(out.cpu(), label.cpu(), None)
                        get_MAP_avg.append(MAP_val)

    #                     MRR = get_MAP_e(out.cpu(), label.cpu(), val_edges)
    #                     get_MRR_avg.append(MRR)

    #                     logging.debug('Epoch: {}, Timestep: {}, Loss: {}, MAP: {}, MRR: {}, Running Mean MAP: {},Running Mean MRR: {}'.format(epoch, ctr, l.item(), get_MAP_e(out.cpu(), label.cpu(), None), MRR,np.asarray(get_MAP_avg).mean(),np.asarray(get_MRR_avg).mean()))

                        MRR = get_MRR(out.cpu(), label.cpu(), np.transpose(a))

                        get_MRR_avg.append(MRR)
            
                        try:
                            if ctr == time_list[time_ctr]:
                                MAP_time.append(MAP_val)
                                MRR_time.append(MRR)
                                time_ctr = time_ctr+1
                        except:
                            pass

                        logging.debug('Epoch: {}, Timestep: {}, Loss: {}, MAP: {}, MRR: {}, Running Mean MAP: {}, Running Mean MRR: {}'.format(epoch, ctr, l.item(), get_MAP_e(out.cpu(), label.cpu(), None),MRR, np.asarray(get_MAP_avg).mean(),np.asarray(get_MRR_avg).mean()))

            A_prev_node = data[ctr][0].shape[0]
            count = count+1



    MAP_l.append(MAP_time)
    MRR_l.append(MRR_time)
    logging.debug('Saving model')
    torch.save(classify.state_dict(), name + '/classifier_'+ str(np.array(mu_64[0]).shape[1])+'.pth')


if not os.path.exists(name+'/saved_array'):
        os.makedirs(name+'/saved_array')
with open(name+'/saved_array/MRR','wb') as f: pickle.dump(MRR_l, f)
with open(name+'/saved_array/MAP','wb') as f: pickle.dump(MAP_l, f)
