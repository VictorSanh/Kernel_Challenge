## Data manipulation
import numpy as np
from numpy.core.defchararray import not_equal

def hamming_distance(source, target):
    """Return the Hamming distance between equal-length sequences"""
    return np.count_nonzero(not_equal(source,target))

def hamming_kernel(source, target):
    """Return the value of a K(s,t) where K is kernel defined from Hamming distance"""
    if len(source) != len(target):
        raise ValueError("Undefined for sequences of unequal length")
    if len(source) == 0:
        raise ValueError("Strings are empty")
    return 1 - (hamming_distance(source, target)*1.0/len(source))

def levenshtein_distance(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]

def levenshtein_kernel(source, target):
    """Return the value of a K(s,t) where K is kernel defined from Hamming distance"""
    if len(source) != len(target):
        raise ValueError("Undefined for sequences of unequal length")
    if len(source) == 0:
        raise ValueError("Strings are empty")
    return 1 - (levenshtein_distance(source, target)*1.0/len(source))