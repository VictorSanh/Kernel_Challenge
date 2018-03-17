## Data manipulation
import numpy as np

## Performance metrics
import time

## Kernel SVM requirements
from cvxopt import matrix
from cvxopt import solvers
solvers.options['show_progress'] = False
import mosek

from scipy.spatial.distance import cdist

## Importing self-made fcts
from metrics import *

## Debugging
import pdb


## General class to build a kernel matrix
def build_kernel(arr1, arr2, kernel_fct, stringsData=True, verbose=True):
    """Builds the kernel matrix from numpy array @arr and kernel function @kernel_fct. V1, unnefficient"""
    try:
        assert len(arr1) > 0
        assert len(arr2) > 0
    except AssertionError:
        print('At least one of the argument arrays is empty')
    if arr1.ndim == 1:
        arr1 = arr1.reshape((len(arr1),1))
    if arr2.ndim == 1:
        arr2 = arr2.reshape((len(arr2),1))
    
    if verbose:
        print('Building kernel matrix from {0:d}x{1:d} samples...'.format(len(arr1),len(arr2)))
    tick = time.time()
    
    if stringsData:
        K = cdist(arr1, arr2, lambda u, v: kernel_fct(u[0],v[0]))
    else:
        K = cdist(arr1, arr2, kernel_fct)
    
    if verbose:
        print('...done in {0:.2f}s'.format(time.time()-tick))
    return K


## 'Default' linear classifier (on numeric data only)
def linear_prod(x1, x2):
    t1 = np.ravel(x1)
    t2 = np.ravel(x2)
    if len(t1) != len(t2):
        raise ValueError("Undefined for sequences of unequal length")
    return np.dot(t1,t2)


## Variable assignement from kwargs
def get_from_KWargs(kwargs, name, default=None):
    if name in kwargs:
        if kwargs[name] is not None:
            return kwargs[name]
    return default

## General method for k-fold cross validation
def kfold(data, labels, n_folds, train_method, pred_method, classify_method, metric, target_folds, **kwargs):
    try:
        assert n_folds > 1
    except AssertionError:
        print('Need more than one fold')

    try:
        assert len(data) == len(labels)
    except AssertionError:
        print('Error: Data and labels have different length')  
    
    if verbose: print('Engaging n-fold cross validation with {0:d} folds on {1:d} items'.format(n_folds, len(data)))    
    fold_size = int(len(data)/n_folds)
    # Random permuation of the data
    perm = np.random.permutation(len(data))
    data = data[perm]
    labels = labels[perm]

    res = []
    for fold in range(n_folds):
        if target_folds is not None and fold not in target_folds:
            res.append(np.nan)
            continue
        val_idx = range(fold*fold_size,(fold+1)*fold_size)
        val_data = np.array(data[val_idx])
        val_labels = np.array(labels[val_idx])

        train_data = np.array([element for i, element in enumerate(data) if i not in val_idx])
        train_labels = np.array([element for i, element in enumerate(labels) if i not in val_idx])

        train_method(train_data, train_labels, **kwargs)

        preds = pred_method(val_data, **kwargs)
        
        if metric.quantized:
            preds = classify_method(preds)
        res.append(metric.measure(np.ravel(preds), val_labels))
        if verbose: print('Fold {0:d}, {1:s}: {2:.2f}'.format(fold,metric.name,res[fold]))

    print('Done! Average {0:s} is {1:.2f}'.format(metric.name,np.nanmean(res)))
    return np.nanmean(res)


###################################
### KERNEL METHODS PARENT CLASS ###
###################################

class kernelMethod():
    def __init__(self):
        pass

    def format_labels(self, labels):
        return labels

    def train(self, data, labels, kernel_fct=None, solver=None, stringsData=True, **kwargs):
        pass

    def predict(self, data, **kwargs):
        pass

    def classify(self, preds):
        return preds

    def assess(self, data, labels, n_folds=1, kernel_fct=None, solver=None, stringsData=True, metric=m_binary, target_folds=None):
        if n_folds > 1:
            return kfold(data, labels, n_folds, self.train, self.predict, self.classify, metric, target_folds, format_labels=self.format_labels, stringsData=stringsData, kernel_fct=kernel_fct, solver=solver)    

    def grid_search(self, data, labels, hyperparameter, search_min, search_max, search_count, n_folds=None, scale='linear', folds_per_search=1, kernel_fct=None):
        try:
            assert search_count > 1
            assert search_max > search_min
            assert folds_per_search > 0
        except AssertionError:
            print('One of arguments provided to grid-search is incorrect')

        grid = []
        total_folds = search_count*folds_per_search
        if n_folds is None:
            n_folds = total_folds

        for it in range(search_count):
            if scale == 'log':
                param = search_min*np.power(search_max*1.0/search_min,it*1.0/(search_count-1))
            else:
                param = search_min + it*1.0/(search_count-1)

            t_folds = np.remainder(range(it*folds_per_search,(it+1)*folds_per_search),n_folds-1)
            hyperparameter = param
            grid.append({'value':param, 'folds':t_folds ,'score':self.assess(data, labels, n_folds, kernel_fct, solver=None, stringsData=False, metric=m_binary, target_folds=t_folds)})

        return grid


##################
### KERNEL SVM ###
##################

class kernelSVM(kernelMethod):
    def __init__(self, lbda=0.1, solver='cvxopt'):
        self.lbda = lbda
        self.solver = solver
        self.data = None
        self.alpha = None
        self.kernel_fct = None
    
    def format_labels(self, labels):
        try:
            assert len(np.unique(labels)) == 2
        except AssertionError:
            print('Error: Labels provided are not binary')
        lm,lM = np.min(labels), np.max(labels)
        l = (labels==lM).astype(int) - (labels==lm).astype(int)
        return l
    
    def train(self, data, labels, **kwargs):
        """Trains the kernel SVM on data and labels"""
        # Default kernel will be linear (only works in for finite-dim floats space)
        kernel_fct = get_from_KWargs(kwargs, 'kernel_fct', linear_prod)
        stringsData = get_from_KWargs(kwargs, 'stringsData', True)
        solver = get_from_KWargs(kwargs, 'solver', 'cvxopt')
        reg = get_from_KWargs(kwargs, 'reg', 0)

        n_samples = labels.shape[0]
        # Turning labels into ±1
        labels = self.format_labels(labels)
        # Binding kernel fct and data as attribute for further predictions
        self.kernel_fct = kernel_fct
        self.data = data
        # Building matrices for solving dual problem
        K = build_kernel(data, data, kernel_fct, stringsData)
        d = np.diag(labels)
        P = matrix(2.0*K + reg*np.eye(n_samples), tc='d')
        q = matrix(-2.0*labels, tc='d')
        G = matrix(np.vstack((-d,d)), tc='d')
        h1 = np.zeros((n_samples,1))
        h2 = (1.0/(2*self.lbda*n_samples))*np.ones((n_samples,1))
        h = matrix(np.vstack((h1,h2)), tc='d')
        # Construct the QP, invoke solver
        sol = solvers.qp(P,q,G,h,solver=solver)
        # Extract optimal value and solution
        self.alpha = np.asarray(sol['x'])
   
    def predict(self, data, **kwargs):
        """Predict labels for data"""
        try:
            assert self.alpha is not None
            assert self.kernel_fct is not None
        except AssertionError:
            print('Error: No successful training recorded')

        stringsData = get_from_KWargs(kwargs, 'stringsData', True)

        # Build sv alpha and sv K(x_i(new_data), x_j(ref))
        sv_ind = np.nonzero(self.alpha)[0]
        sv_alpha = self.alpha[sv_ind]
        sv_K = build_kernel(data, self.data[sv_ind], self.kernel_fct, stringsData)
        # Use supvec alpha and supvec K to compute predictions
        return sv_K @ sv_alpha

    def classify(self, preds):
            return self.format_labels(preds)

    def grid_search(self, data, labels, search_min, search_max, search_count, n_folds=None, scale='linear', folds_per_search=1, kernel_fct=None):
        return super().grid_search(data, labels, self.lbda, search_min, search_max, search_count, n_folds, scale, folds_per_search, kernel_fct)


##################
### KERNEL kNN ###
##################

class kernelKNN(kernelMethod):
    def __init__(self, k):
        self.k = k

    def train(self, data, labels, **kwargs):
        self.ref_data = data
        self.ref_labels = labels

    def predict(self, data, **kwargs):
        ##  first let's find kNN for all points in the dataset
        kernel_fct = get_from_KWargs(kwargs, 'kernel_fct', linear_prod)

        self.kernel_fct = kernel_fct
        K = build_kernel(data, self.ref_data, self.kernel_fct, stringsData=False)
        idx = (np.argsort(K)[:,-self.k:])
        labels = np.array(self.ref_labels)[idx]
        bincount = np.apply_along_axis(lambda x: np.bincount(x, minlength=2), axis=1, arr=labels)
        return np.argmax(bincount, axis=1)

    def grid_search(self, data, labels, search_min, search_max, search_count, n_folds=None, scale='linear', folds_per_search=1, kernel_fct=None):
        return super().grid_search(data, labels, self.k, search_min, search_max, search_count, n_folds, scale, folds_per_search, kernel_fct)