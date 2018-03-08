## Data manipulation
import numpy as np
import pdb

## Squared loss
def ls_squared(preds, labels):
    """Returns the hinge loss for preds %labels"""
    try:
        assert len(preds) == len(labels)
    except AssertionError:
        print('Error: preds and labels have different lengths')
    n_samples = len(preds)
    return (1.0*np.power(np.linalg.norm(preds-labels),2)) / n_samples

## 0/1 loss
def ls_binary(preds, labels):
    """Returns the 0/1 loss for preds %labels"""
    preds = np.sign(preds)
    return ls_squared(0.5*preds, 0.5*labels)

## Hinge loss
def ls_hinge(preds, labels):
    """Returns the hinge loss for preds %labels"""
    try:
        assert len(preds) == len(labels)
    except AssertionError:
        print('Error: reds and labels have different lengths')
    n_samples = len(preds)
    return np.mean(
        np.maximum(
            np.ones((n_samples,1)) - preds*labels,
            np.zeros((n_samples,1))
        )
    )

## Building metrics for reporting on performance
class Metric():
    def __init__(self, name, measure):
        self.name = name
        self.measure = measure
        
m_binary = Metric('Match rate', lambda preds,labels: 1 - ls_binary(preds,labels))