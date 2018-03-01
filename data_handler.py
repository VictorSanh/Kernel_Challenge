import numpy as np
import pandas as pd

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


def format_preds(preds):
    return (0.5*(1+np.sign(preds))).astype(int)