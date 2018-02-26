## Data manipulation
import numpy as np

## Performance metrics
import time

## Kernel SVM requirements
from cvxopt import matrix
from cvxopt import solvers
from scipy.spatial.distance import cdist
from numpy.core.defchararray import not_equal


def build_kernel(arr1, arr2, kernel_fct):
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
    K = cdist(arr1, arr2, lambda u, v: kernel_fct(list(u[0]),list(v[0])))
    return K


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
    
    def run(self, data, labels, kernel_fct):
        """Trains the kernel SVM on data and labels"""
        n_samples = labels.shape[0]
        # Turning labels into ±1
        labels = self.format_labels(labels)
        # Binding kernel fct and data as attribute for further predictions
        self.kernel_fct = kernel_fct
        self.data = data
        # Building matrices for solving dual problem
        print('Building kernel matrix from {0:d} samples...'.format(n_samples))
        tick = time.time()
        K = build_kernel(data, data, kernel_fct)
        print('...done in {0:.2f}s'.format(time.time()-tick))
        d = np.diag(labels)
        P = matrix((-1.0/(2*self.lbda))*d*K*d, tc='d')
        q = matrix(np.ones((n_samples,1)), tc='d')
        G1 = -np.eye(n_samples)
        G2 = np.eye(n_samples)
        G = matrix(np.vstack((G1,G2)), tc='d')
        h1 = np.zeros((n_samples,1))
        h2 = (1.0/n_samples)*np.ones((n_samples,1))
        h = matrix(np.vstack((h1,h2)), tc='d')
        # Construct the QP, invoke solver
        sol = solvers.qp(P,q,G,h)
        # Extract optimal value and solution
        dual = sol['x']
        # Solving dual problem via solver
        self.alpha = (1.0/(2*self.lbda))*(d @ dual)
    
    def predict(self, data):
        """Predict labels for data"""
        try:
            assert self.alpha is not None
            assert self.kernel_fct is not None
        except AssertionError:
            print('Error: No successful training recorded')
        # Build sv alpha and sv K(x_i(new_data), x_j(ref))
        sv_ind = np.nonzero(self.alpha)[0]
        sv_alpha = self.alpha[sv_ind]
        sv_K = build_kernel(data, self.data[sv_ind], self.kernel_fct)
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