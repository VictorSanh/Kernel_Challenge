## Data manipulation
import numpy as np
from numpy.core.defchararray import not_equal

###################
##### HAMMING #####
def hamming_distance(source, target):
    """Return the Hamming distance between equal-length sequences"""
    return np.count_nonzero(not_equal(source,target))

def hamming_kernel(source, target):
    """Return the value of a K(s,t) where K is kernel defined from Hamming distance"""
    source = list(source)
    target = list(target)
    if len(source) != len(target):
        raise ValueError("Undefined for sequences of unequal length")
    if len(source) == 0:
        raise ValueError("Strings are empty")
    return 1 - (hamming_distance(source, target)*1.0/len(source))


#######################
##### LEVENSHTEIN #####
def levenshtein_distance(source, target):
    source = list(source)
    target = list(target)
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


####################
##### SPECTRUM #####
def create_dictionary(training_sequences, substring_length):
    '''Create the dictionary/vocabulary of possible subsequeces of length substring_length from training sequences.
    "ABCD" contains two words of legnth 3: "ABC" and "BCD".
    
    Input:
        training_sequences: array like strucutre containing training sequences
        substring_length: length of substring in sequence
    Output:
        word_2_index: mapping between a word and its index. The keys are all the words of length substring_length appearing in training sequences. 
    '''
    
    unique_subsequences = set()

    for string in training_sequences:
        for start in range(len(string)-substring_length+1):
            end = start + substring_length
            substring = string[start:end]
            unique_subsequences.add(substring)
    
    #Creating the word_2_index mapping words and their index. The keys are all the words.
    unique_subsequences = sorted(unique_subsequences)
    word_2_index = dict()
    for idx, word in enumerate(unique_subsequences):
        word_2_index[word] = idx
        
    return word_2_index

def create_occ_feature(sequence, substring_length, dictionary, normalize=True):
    '''Create the spectrum kernel feature vector of occurences of every word in dictionary/vocabulary.
    
    Input:
        sequence: ADN sequence to transform
        dictionary: already trained dictionary listing all the words appearing in training and their index
        normalize: if true, transform the occurences in percentage (frequencies)
    Ouput:
        feature: occurence of each word in dictionary/vocabulary
    '''
    
    feature = np.zeros(len(dictionary), dtype = int)
    
    for start in range(len(sequence)-substring_length+1):
        end = start + substring_length
        substring = sequence[start:end]
        if substring in dictionary: #It is possible that some word in test are not appearing in training
            feature[dictionary[substring]] = feature[dictionary[substring]] + 1
            
    if normalize:
        feature = feature/feature.sum()
        
    return feature

def spetrum_kernel(sequence_A, sequence_B, substring_length, dictionary, normalize=False):
    '''substring_length-spectrum kernel
    
    Input:
        sequence_A: first sequence
        sequence_B: second sequence
        substring_length: length of word in vocabulary
        dictionary: vocabulary derived from training
        normalize: if true, transform the occurences in percentage (frequencies)
    Output:
        kernel similarity between sequence_A and sequence_B
    '''
    
    feature_A = create_occ_feature(sequence_A, substring_length, dictionary, normalize)
    feature_B = create_occ_feature(sequence_B, substring_length, dictionary, normalize)
    
    return np.dot(feature_A, feature_B)