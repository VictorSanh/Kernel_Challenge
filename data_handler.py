import numpy as np
import pandas as pd

###############################
# Data loading and formatting #
###############################

def load_data(dsID, set_type='tr', folder_name='data'):
    Xdata_file = folder_name + '/X' + set_type + str(dsID) + '.csv'
    X = pd.read_csv(Xdata_file, header=None, names=['Sequence'], dtype={'Sequence': np.unicode_})
    if set_type=='tr':
        Ydata_file = folder_name + '/Y' + set_type + str(dsID) + '.csv'
        Y = pd.read_csv(Ydata_file, index_col=0, dtype={'Bound': np.dtype(bool)})
        Y.index = Y.index - 1000*dsID
        df = pd.concat([X, Y], axis=1)
    else:
        df = X
    return df

def load_datafeats(dsID, set_type='tr', folder_name='data'):
    Xdata_file = folder_name + '/X' + set_type + str(dsID) + '_mat50.csv'
    df = pd.read_csv(Xdata_file, header=None, sep=" ")
    return df.as_matrix()

def format_preds(preds):
    return (0.5*(1+np.sign(preds))).astype(int)


def data_normalization(data, offset_column=False):
    d_mean = np.mean(data, axis=0)
    d_std = np.std(data, axis=0)
    data = (data - d_mean)/d_std
    if offset_column:
        data = np.hstack((data,np.ones((len(data),1))))
    return data

#####################################
# Weighting different classifiers   #
# to potentially do better than all #
#####################################

def voting(preds, wghts, stochastic=False):
    votes =  np.average(preds, axis=1, weights=wghts)
    if stochastic:
    	return np.random.binomial(1, p=votes).astype(int)
    else:
    	return (0.5*(1 + np.sign(votes-0.5))).astype(int)


##########################################
# Mutual-information based dim reduction #
##########################################

def get_MI(data, labels, word_idx, bins):
    n,p = data.shape
    idx_bound = np.argwhere(labels==1)
    idx_unbound = np.argwhere(labels==0)
    data_bound = np.take(data, idx_bound, axis=0)
    data_unbound = np.take(data, idx_unbound, axis=0)
    
    n_b = len(data_bound)
    n_ub = n - n_b
    data_bound = data_bound.reshape((n_b,p))
    data_unbound = data_unbound.reshape((n_ub,p))
    
    p_b = n_b*1.0/n
    p_ub = 1.0 - p_b
    
    MI = 0
    for abin in bins:
        b_cond = np.count_nonzero(np.isin(data_bound[:,word_idx], abin))*1.0/n_b
        ub_cond= np.count_nonzero(np.isin(data_unbound[:,word_idx], abin))*1.0/n_ub

        cond_data = np.isin(data[:,word_idx], abin)
        n_cond = np.count_nonzero(cond_data)
        if n_cond == 0:
            continue
        cond_b = np.count_nonzero(labels[cond_data]==1)*1.0/n_cond
        cond_ub = 1.0 - cond_b       
    
        if cond_b > 0:
            MI = MI + b_cond*p_b*np.log(cond_b/p_b)
        if cond_ub > 0:
            MI = MI + ub_cond*p_ub*np.log(cond_ub/p_ub)
        if np.isnan(MI):
            pdb.set_trace()
    return MI

def argmax_MI(data, labels, n_feats, bins):
    n,p = data.shape
    MI = np.zeros(p)
    for word_idx in range(p):
        MI[word_idx] = get_MI(data, labels, word_idx, bins)
    max_MI_idx = np.argsort(MI)[-1:-(n_feats+2):-1]
    return max_MI_idx, MI[max_MI_idx]

def MI_dimRed(data, labels, n_feats, bins):
    idx, MI_ranked = argmax_MI(data, features, n_feats, bins)
    data_lowdim = np.take(data, idx, axis=1)
    return data_lowdim, idx, MI_ranked