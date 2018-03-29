# -*- coding: utf-8 -*-

import numpy as np
import itertools

from kernel_methods import *
from kernels import *
from data_handler import *
from LogisticRegression import *





if __name__ == '__main__':

	#### DATASET 0 ####
	###################
	print('\n\nHandling Dataset 0')

	#Load Data
	data_train = load_data(0, 'tr')
	Y_train = data_train['Bound'].as_matrix()
	data_test = load_data(0, 'te')

	#Define hyperparameters
	substring_length = 6
	dictionary = create_dictionary(data_train['Sequence'], substring_length)

	#Create spectrum features
	X_train = np.apply_along_axis(lambda seq_A: create_occ_feature(seq_A[0], substring_length, dictionary, normalize=False), axis=1, arr=data_train['Sequence'].as_matrix().reshape(2000,1))
	X_test = np.apply_along_axis(lambda seq_A: create_occ_feature(seq_A[0], substring_length, dictionary, normalize=False), axis=1, arr=data_test['Sequence'].as_matrix().reshape(1000,1))
	
	#Train
	#regu = 0.01
	#kLGR = kernelLogisticRegression(regu)
	#kLGR.train(X_train, Y_train, kernel_fct = linear_prod, stringsData = False)
	logreg = LogisticRegression(lbda = 150)
	logreg.train(X_train, Y_train)
	
	#Predict
	#preds = kLGR.predict(X_test, kernel_fct = linear_prod, stringsData = False)
	#te0_raw = kLGR.classify(preds)
	te0_raw = logreg.predict(X_test)
		
	
		
	#### DATASET 1 ####
	###################
	print('\n\nHandling Dataset 1')
	
	#Load Data
	data_train = load_data(1, 'tr')
	Y_train = data_train['Bound'].as_matrix()
	data_test = load_data(1, 'te')
	
	#Define hyperparameters
	alphabet = ['A', 'C', 'G', 'T']
	substring_length = 7
	mismatch_tol = 1
	vocab2index, _ = create_vocab(alphabet, substring_length)
	neighbours = compute_neighbours(vocab2index, mismatch_tol)
	
	#Create mismatch features
	X_train = np.apply_along_axis(lambda seq_A: create_mismatch_feature(seq_A[0], substring_length, vocab2index, neighbours, normalize=False), axis=1, arr=data_train['Sequence'].as_matrix().reshape(2000,1))
	X_test = np.apply_along_axis(lambda seq_A: create_mismatch_feature(seq_A[0], substring_length, vocab2index, neighbours, normalize=False), axis=1, arr=data_test['Sequence'].as_matrix().reshape(1000,1))
	
	#Train
	lbda = 1
	kSVM = kernelSVM(lbda)
	kSVM.train(X_train, Y_train, kernel_fct = linear_prod, stringsData = False)

	#Predict
	preds = kSVM.predict(X_test, kernel_fct = linear_prod, stringsData = False)
	te1_raw = kSVM.classify(preds)
	
	
	
	
	#### DATASET 2 ####
	###################
	print('\n\nHandling Dataset 2')
	
	#Load Data
	data_train = load_data(2, 'tr')
	Y_train = data_train['Bound'].as_matrix()
	data_test = load_data(2, 'te')
	
	#Define hyperparameters
	alphabet = ['A', 'C', 'G', 'T']
	substring_length = 5
	mismatch_tol = 1
	vocab2index, _ = create_vocab(alphabet, substring_length)
	neighbours = compute_neighbours(vocab2index, mismatch_tol)
	
	#Create mismatch features
	X_train = np.apply_along_axis(lambda seq_A: create_mismatch_feature(seq_A[0], substring_length, vocab2index, neighbours, normalize=False), axis=1, arr=data_train['Sequence'].as_matrix().reshape(2000,1))
	X_test = np.apply_along_axis(lambda seq_A: create_mismatch_feature(seq_A[0], substring_length, vocab2index, neighbours, normalize=False), axis=1, arr=data_test['Sequence'].as_matrix().reshape(1000,1))
		
	#Train
	lbda = 1
	kSVM = kernelSVM(lbda)
	kSVM.train(X_train, Y_train, kernel_fct = linear_prod, stringsData = False)
	
	#Predict
	preds = kSVM.predict(X_test, kernel_fct = linear_prod, stringsData = False)
	te2_raw = kSVM.classify(preds)
	
	
	
	
	#### Concatenate predictions ####
	#################################
	print('\n\nConcatenate predictions')
	
	te0_raw = pd.DataFrame(
		data = format_preds(te0_raw),
		columns = ['Bound'])

	te1_raw = pd.DataFrame(
		data = format_preds(te1_raw),
		columns = ['Bound'])
	te1_raw.index = te1_raw.index + 1000

	te2_raw = pd.DataFrame(
		data = format_preds(te2_raw),
		columns = ['Bound'])
	te2_raw.index = te2_raw.index + 2000

	frames = [te0_raw, te1_raw, te2_raw]
	te = pd.concat(frames)
	te.index = te.index.set_names(['Id'])

	te.to_csv('predictions/Yte.csv')
