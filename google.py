from data_handler import *

import numpy as np
import pdb
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

from data_handler import *
from kernels import *
from kernel_methods import *


tr2 = load_data(2, 'tr')

substring = substringKernel(subseq_length=5, lambda_decay=0.5)
kSVM = kernelSVM()
kSVM.assess(tr2['Sequence'].as_matrix(), 
           tr2['Bound'].as_matrix(),
           n_folds=5,
           kernel_fct = lambda seq_A, seq_B: substring._K(5, seq_A, seq_B),
           stringsData=True)