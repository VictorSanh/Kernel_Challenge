## Data manipulation
import numpy as np

## Performance metrics
import time

## Kernel SVM requirements
from cvxopt import matrix
from cvxopt import solvers
import mosek

from scipy.spatial.distance import cdist


## General class to build a kernel matrix
def build_kernel(arr1, arr2, kernel_fct, stringsData=True):
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
    if stringsData:
        K = cdist(arr1, arr2, lambda u, v: kernel_fct(list(u[0]),list(v[0])))
    else:
        K = cdist(arr1, arr2, kernel_fct)
    return K


## 'Default' linear classifier (on numeric data only)
def linear_prod(x1, x2):
    t1 = np.ravel(x1)
    t2 = np.ravel(x2)
    if len(t1) != len(t2):
        raise ValueError("Undefined for sequences of unequal length")
    return np.dot(t1,t2)


## Kernel methods parent class
class kernelMethod():
    def __init__(self):
        return 0


## Kernel SVM method
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
    
    def run(self, data, labels, kernel_fct=None, solver=None, stringsData=True):
        """Trains the kernel SVM on data and labels"""
        # Default kernel will be linear (only works in for finite-dim floats space)
        if kernel_fct is None:
            kernel_fct = linear_prod
        n_samples = labels.shape[0]
        # Turning labels into ±1
        labels = self.format_labels(labels)
        # Binding kernel fct and data as attribute for further predictions
        self.kernel_fct = kernel_fct
        self.data = data
        # Building matrices for solving dual problem
        print('Building kernel matrix from {0:d} samples...'.format(n_samples))
        tick = time.time()
        K = build_kernel(data, data, kernel_fct, stringsData) 
        print('...done in {0:.2f}s'.format(time.time()-tick))
        d = np.diag(labels)
        P = matrix(2.0*K, tc='d')
        q = matrix(-2.0*labels, tc='d')
        G = matrix(np.vstack((-d,d)), tc='d')
        h1 = np.zeros((n_samples,1))
        h2 = (1.0/(2*self.lbda*n_samples))*np.ones((n_samples,1))
        h = matrix(np.vstack((h1,h2)), tc='d')
        # Construct the QP, invoke solver
        sol = solvers.qp(P,q,G,h,solver=solver)
        # Extract optimal value and solution
        self.alpha = np.asarray(sol['x'])
    
    def predict(self, data, stringsData=True):
        """Predict labels for data"""
        try:
            assert self.alpha is not None
            assert self.kernel_fct is not None
        except AssertionError:
            print('Error: No successful training recorded')
        # Build sv alpha and sv K(x_i(new_data), x_j(ref))
        sv_ind = np.nonzero(self.alpha)[0]
        sv_alpha = self.alpha[sv_ind]
        sv_K = build_kernel(data, self.data[sv_ind], self.kernel_fct, stringsData)
        # Use supvec alpha and supvec K to compute predictions
        return sv_K @ sv_alpha
    
    def assess(self, data, labels, metrics):
        """Provides the performance of the algorithm on some test data"""
        try:
            assert len(data) == len(labels)
        except AssertionError:
            print('Error: Data and labels have different length')
        labels = self.format_labels(labels).reshape((len(labels),1))
        preds = self.predict(data)
        m = {}
        if metrics is not None:
            for metric in metrics:
                m[metric.name] = metric.measure(preds, labels)
        return preds, m