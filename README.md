# Kernel_Challenge
MVA - Kernek Methods in Machine Learning - Data Challenge 
_________________________________________________________


#### How to install the requirements ?
pip3 install -r requirements.txt


##### PACKAGES - NOTES #####

## Special instructions for Mosek
Mosek is a powerful optimization package:
https://www.mosek.com/
A free academic license can be downloaded at:
https://www.mosek.com/products/academic-licenses/
The package provides useful feedback on optimization performance and detailed debugging feedback.

## How to use k-fold cross-validation

- If you are using a kernelMethod class (for instance kernelSVM or kernelKNN), just use method assess():
Ex: kNN.assess(data, labels, n_folds, stringsData)
	- data and labels are given as matrices (tr0['Sequence'].as_matrix() will do the trick)
	- n_folds is the number of folds for cross-valisation
	- stringsData is True if you are providing the original text data, False if you are passing transformed data that lives in a Euclidian space

- If you are not using a kernelMethod class, use fonction kfold() from kernel_methods.py:
Ex: kfold(data, labels, n_folds, train_method, predict_method, metric, **kwargs)
	- train_method is what will be used to train your model from train data (arg1) and train labels (arg2). It should be a method from your classifier that modifies learned parameters stored as classifier attributes. **kwargs will be passed to this method
	- pred_method is the method for generating prediction from validation data (arg1). **kwargs will be passed to this method
	- metric will measure the performance by comparing labels to predictions. For instance, a m_binary from metrics.py