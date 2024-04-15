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
import json
warnings.filterwarnings('ignore')




from copy import deepcopy



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

def init_logging_handler(exp_name):
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('{}/log_{}.txt'.format(exp_name, current_time))
    logging.basicConfig(handlers=[stderr_handler, file_handler])
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

name = 'Results/AS'
init_logging_handler(name)
logging.debug(str(config))

def check_if_gpu():
    if torch.cuda.is_available():
        device = 'cuda:0'
#         device = 'cpu'

    else:
        device = 'cpu'
    return device
device = check_if_gpu()
#device = 'cpu'
logging.debug('The code will be running on {}'.format(device))


def is_compatible(filename):
    return any(filename.endswith(extension) for extension in ['.txt'])


class dataset_as(torch.utils.data.Dataset):
    def __init__(self, root_dir, train = True):
      
        self.graph_root_dir = root_dir
        
        self.graph_filename = [join(self.graph_root_dir, x) for x in listdir(self.graph_root_dir) if is_compatible(x)]
        self.graph_filename.sort()
        self.graph_filename = self.graph_filename[0:100]
        self.Adj_arr = []
        self.X_Sparse_arr = []
        count = 0
        max_size = 0
        for filename in self.graph_filename:
            logging.debug(count)
            A, X_Sparse, size = self.get_graph(filename, max_size)
            if size > max_size:
                max_size = size
                
            self.Adj_arr.append(A)
            self.X_Sparse_arr.append(X_Sparse)
            count = count + 1
        
 
      
    def __len__(self):
        return len(self.graph_filename)
    
    
    def __getitem__(self, idx):
        return self.Adj_arr[idx], self.X_Sparse_arr[idx]
    
    
    def get_graph(self, filename, max_size):
        emp = []
        with open(filename) as fp:            
            line = fp.readline()
            while line:
                emp.append(line.strip())
                line = fp.readline()
        a = emp[4:]
        new_a = []
        for i in a:
            new_a.append(i.split('\t'))
        if max_size > int(new_a[-1][0]):
            new_max = max_size
            arr_zero = np.zeros((max_size, max_size))
        else:
            arr_zero = np.zeros((int(new_a[-1][0]), int(new_a[-1][0])))
            new_max =  int(new_a[-1][0])
        
        row_ind = []
        col_ind = []
        data    = []

        for i in new_a:
            row_ind.append(int(i[0]))
            col_ind.append(int(i[1]))
            data.append(1)
            #arr_zero[int(i[0])-1][int(i[1])-1] = 1 
        #arr_zero[range(len(arr_zero)), range(len(arr_zero))] = 0
        #A = csr_matrix(arr_zero)

        data = np.array(data)
        row_ind = np.array(row_ind) - 1
        col_ind = np.array(col_ind) - 1
        A = csr_matrix((data, (row_ind, col_ind)), shape = (new_max, new_max))
        X= A + sp.eye(A.shape[0])
        X_Sparse = sparse_feeder(X)
        X_Sparse = spy_sparse2torch_sparse(X_Sparse)
        
        return A, X_Sparse, new_max


data = dataset_as('../../../datasets/as_data')


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
    


import pickle



name_loaded = 'Results/AS'
with open(name_loaded+'/Eval_Results/saved_array/mu_as','rb') as f: mu_arr = pickle.load(f)
with open(name_loaded+'/Eval_Results/saved_array/sigma_as','rb') as f: sigma_arr = pickle.load(f)
    

MAP_l = []
MRR_l = []

time_list = [80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99]

# In[7]:
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


    seed = 5
    torch.cuda.manual_seed_all(seed)
    classify = Classifier()
    classify.to(device)
    
    loss = torch.nn.BCEWithLogitsLoss(reduce=False)

    optim = torch.optim.Adam(classify.parameters(), lr = 1e-3)
    
    #Pre-computing pos and neg samples
    print("Precomputing positive and negative samples")
    val_edges_dict = {}
    count = 0
    for ctr,i in enumerate(data):
        A_node = i[0].shape[0]
        A = i[0]
        if count>0:
            if A_node>A_prev_node:
                A = A[:A_prev_node,:A_prev_node]
            if ctr<70:
                ones_edj  = A.nnz
                zeroes_edj= A.shape[0]*100
                tot = ones_edj + zeroes_edj
                val_ones = list(set(zip(*A.nonzero())))
                val_ones = random.sample(val_ones, ones_edj)
                val_ones = [list(ele) for ele in val_ones]

                val_zeros = sample_zero_n(A,zeroes_edj)
                val_zeros = [list(ele) for ele in val_zeros]
                val_edges = np.row_stack((val_ones,val_zeros))
                val_edges_dict[ctr] = val_edges
        print("Pre-computing ts:",ctr)
        A_prev_node = i[0].shape[0]
        count = count+1
 
    num_epochs = 10
    for epoch in range (num_epochs):
            count = 0
            for ctr in range(lookback+1,70):

                A_node = data[ctr][0].shape[0]
                A = data[ctr][0]

                if count > 0:
                    if A_node > A_prev_node:
                        A = A[:A_prev_node,:A_prev_node]

                    if ctr < 70:
                        logging.debug('Training')
                        logging.debug(ctr)

                        ones_edj = A.nnz
                        zeroes_edj = A.shape[0]*100
                        #zeroes_edj = A.nnz
                        tot = ones_edj + zeroes_edj

                        #val_ones = list(set(zip(*A.nonzero())))
                        #val_ones = random.sample(val_ones, ones_edj)
                        #val_ones = [list(ele) for ele in val_ones] 
                        #val_zeros = sample_zero_n(A,zeroes_edj)
                        #val_zeros = [list(ele) for ele in val_zeros] 
                        #val_edges = np.row_stack((val_ones, val_zeros))
                        
                        val_edges = val_edges_dict[ctr]

                        val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1

                        a, b = unison_shuffled_copies(val_edges,val_ground_truth, count)     

                        if ctr > 0:
                            a_embed = np.array(mu_64[ctr-lookback-1])[a.astype(int)]
                            a_embed_sig = np.array(sigma_64[ctr-lookback-1])[a.astype(int)]


                            classify.train()

                            inp_clf = []
                            for d_id in range (tot):
                                inp_clf.append(np.concatenate((a_embed[d_id][0], a_embed[d_id][1]), axis = 0))

                            inp_clf = torch.tensor(np.asarray(inp_clf))

                            inp_clf = inp_clf.to(device)
                            out = classify(inp_clf).squeeze()

                            weight = torch.tensor([0.1, 0.9]).to(device)

                            label = torch.tensor(np.asarray(b)).to(device)

                            weight_ = weight[label.data.view(-1).long()].view_as(label)

                            l = loss(out, label.float())
                            out = out.cpu()
                            label = label.cpu()

                            l = l  * weight_
                            l = l.mean()

                            optim.zero_grad()

                            l.backward()
                            optim.step()

                            MRR = get_MRR(out.cpu(), label.cpu(), np.transpose(a))

                            logging.debug('L:{}, Epoch: {}, Timestep: {}, Loss: {}, MAP: {}, MRR: {}'.format(np.array(mu_64[0]).shape[1],epoch, ctr, l.item(), get_MAP_e(out.cpu(), label.cpu(), None),MRR))

                A_prev_node = data[ctr][0].shape[0]
                count = count+1

    num_epochs = 1
    MAP_time = []
    MRR_time = []
    time_ctr = 0
    for epoch in range (num_epochs):
        get_MAP_avg = []
        get_MRR_avg = []
        count = 0
        for ctr in range(80,100):

            A_node = data[ctr][0].shape[0]
            A = data[ctr][0]

            if count > 0:
                if A_node > A_prev_node:
                    A = A[:A_prev_node,:A_prev_node]

                if ctr >= 80:
                    logging.debug('Testing')
                    logging.debug(ctr)

                    ones_edj = A.nnz
                    zeroes_edj = A.shape[0]*100
                    #zeroes_adj = A.nnz
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

                        a_embed = np.array(mu_64[ctr-lookback-1])[a.astype(int)]
                        a_embed_sig = np.array(sigma_64[ctr-lookback-1])[a.astype(int)]

                        classify.eval()

                        inp_clf = []
                        for d_id in range (tot):
                            inp_clf.append(np.concatenate((a_embed[d_id][0], a_embed[d_id][1]), axis = 0))

                        inp_clf = torch.tensor(np.asarray(inp_clf))

                        inp_clf = inp_clf.to(device)
                        with torch.no_grad():
                            out = classify(inp_clf).squeeze()

                        label = torch.tensor(np.asarray(b)).to(device)

                        weight = torch.tensor([0.1, 0.9]).to(device)
                        #weight = torch.tensor([0.5,0.5]).to(device)

                        weight_ = weight[label.data.view(-1).long()].view_as(label)

                        l = loss(out, label.float())

                        l = l  * weight_
                        l = l.mean()


                        out = out.cpu()
                        label = label.cpu()





                        MAP_val =  get_MAP_e(out.cpu(), label.cpu(), None)
                        get_MAP_avg.append(MAP_val)


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
#     torch.save(classify.state_dict(), name + '/new_classifier_'+ str(np.array(mu_64[0]).shape[1])+'.pth')
    torch.save(classify.state_dict(), name + '/classifier_'+ str(np.array(mu_64[0]).shape[1])+'.pth')



                                           
if not os.path.exists(name+'/saved_array'):
        os.makedirs(name+'/saved_array')
with open(name+'/saved_array/MRR','wb') as f: pickle.dump(MRR_l, f)
with open(name+'/saved_array/MAP','wb') as f: pickle.dump(MAP_l, f)






