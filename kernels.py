## Data manipulation
import numpy as np
from numpy.core.defchararray import not_equal
from functools import wraps
import itertools

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
    
    
####################
##### MISMATCH #####
def create_vocab(alphabet, substring_length):
    '''
    Create all the vocabulary of all possibles words using the alphabet: all
    combination of length substring_length. Vocabulary is of size |alphabet|^substring_length.
    
    Input:
        alphabet: letters available in the alphabet
        substring_length: lenghth of words
        
    Output:
        vocab2index: dictionary associating each word in the vocab to an index (integer)
        index2vocab: dictionary associating each index to a word in the vocab
    '''
    vocab = [''.join(i) for i in itertools.product(alphabet, repeat = substring_length)]
    
    vocab2index = {}
    index2vocab = {}
    for idx, v in enumerate(vocab):
        vocab2index[v] = idx
        index2vocab[idx] = v
        
    return vocab2index, index2vocab


def is_neighbour(alpha, beta, mismatch):
    '''
    Check if word beta is in the neighbourhood of word alpha as defined by Leslie and al.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.86.7384&rep=rep1&type=pdf
    
    Input:
        alpha: first word
        beta: second word
        mismatch: tolerance of mismatch
    Output
        Boolean: True if beta is the mismatch-neighbourhood of alpha
    '''
    if sum(a!=b for a, b in zip(alpha, beta)) <= mismatch:
        return True
    else:
        return False
    
def compute_neighbours(vocab2index, mismatch):
    '''
    Compute once for all the neighbours of each word in the vocabulary.
    
    Input:
        vocab2index: vocabulary
        mismatch: tolerance of mismatch
    Output:
        Dictionary of neighbours for each word in the vocabulary.
    '''
    vocab = vocab2index.keys()
    
    neighbours = {}
    for word1 in vocab:
        neighbours[word1] = []
        for word2 in vocab:
            if is_neighbour(word1, word2, mismatch):
                neighbours[word1].append(word2)
    
    return neighbours


def create_mismatch_feature(sequence, substring_length, vocab2index, neighbours, normalize = False):
    '''
    Mismatch kernel feature as described by Leslie and al.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.86.7384&rep=rep1&type=pdf
    
    Input:
        sequence: DNA sequence to process
        substring_length: lenghth of vocabulary words
        vocab2index: mapping of vocabulary word to their index
        neighbours: neighbours for each word for each of the word in the vocabulary
        normalize: norm normalization if True
    Output:
        Numpy array: Sequence embedding
    '''
    embedding = np.zeros(len(vocab2index), dtype = 'int')

    for start in range(len(sequence) - substring_length + 1):
        end = start + substring_length
        substring = sequence[start:end]
        for neighbour in neighbours[substring]:
            embedding[vocab2index[neighbour]] += 1
    
    if normalize:
        embedding = embedding/np.linalg.norm(embedding)
        
    return embedding


def mismatch_kernel(sequenceA, sequenceB, substring_length, vocab2index, neighbours, normalize):
    '''
    Mismatch kernel. Optional normalization as described in Leslie and al.
    '''
    embedingA = create_mismatch_feature(sequenceA, substring_length, vocab2index, neighbours, normalize)
    embedingB = create_mismatch_feature(sequenceB, substring_length, vocab2index, neighbours, normalize)
    
    return np.dot(embedingA, embedingB)



#####################
##### SUBSTRING #####

def caching():
    """
    Cache decorator. Arguments to the cached function must be hashable.
    """
    def decorate_func(func):
        cache = dict()
        # separating positional and keyword args
        kwarg_point = object()

        @wraps(func)
        def cache_value(*args, **kwargs):
            key = args
            if kwargs:
                key += (kwarg_point,) + tuple(sorted(kwargs.items()))
            if key in cache:
                result = cache[key]
            else:
                result = func(*args, **kwargs)
                cache[key] = result
            return result

        def cache_clear():
            """
            Clear the cache
            """
            cache.clear()

        # Clear the cache
        cache_value.cache_clear = cache_clear
        return cache_value
    return decorate_func


class substringKernel():
    def __init__(self, subseq_length=3, lambda_decay=0.5):
        """
        Constructor
        :param lambda_decay: lambda parameter for the algorithm
        :type  lambda_decay: float
        :param subseq_length: maximal subsequence length
        :type subseq_length: int
        """
        self.lambda_decay = lambda_decay
        self.subseq_length = subseq_length


    @caching()
    def _K(self, n, s, t):
        """
        K_n(s,t) in the original article; recursive function
        :param n: length of subsequence
        :type n: int
        :param s: document #1
        :type s: str
        :param t: document #2
        :type t: str
        :return: float value for similarity between s and t
        """
        if min(len(s), len(t)) < n:
            return 0
        else:
            part_sum = 0
            for j in range(1, len(t)):
                if t[j] == s[-1]:
                    #not t[:j-1] as in the article but t[:j] because of Python slicing rules!!!
                    part_sum += self._K1(n - 1, s[:-1], t[:j])
            result = self._K(n, s[:-1], t) + self.lambda_decay ** 2 * part_sum
            return result


    @caching()
    def _K1(self, n, s, t):
        """
        K'_n(s,t) in the original article; auxiliary intermediate function; recursive function
        :param n: length of subsequence
        :type n: int
        :param s: document #1
        :type s: str
        :param t: document #2
        :type t: str
        :return: intermediate float value
        """
        if n == 0:
            return 1
        elif min(len(s), len(t)) < n:
            return 0
        else:
            part_sum = 0
            for j in range(1, len(t)):
                if t[j] == s[-1]:
        #not t[:j-1] as in the article but t[:j] because of Python slicing rules!!!
                    part_sum += self._K1(n - 1, s[:-1], t[:j]) * (self.lambda_decay ** (len(t) - (j + 1) + 2))
            result = self.lambda_decay * self._K1(n, s[:-1], t) + part_sum
            return result


    def _gram_matrix_element(self, s, t, sdkvalue1, sdkvalue2):
        """
        Helper function
        :param s: document #1
        :type s: str
        :param t: document #2
        :type t: str
        :param sdkvalue1: K(s,s) from the article
        :type sdkvalue1: float
        :param sdkvalue2: K(t,t) from the article
        :type sdkvalue2: float
        :return: value for the (i, j) element from Gram matrix
        """
        if s == t:
            return 1
        else:
            try:
                return self._K(self.subseq_length, s, t) / \
                       (sdkvalue1 * sdkvalue2) ** 0.5
            except ZeroDivisionError:
                print("Maximal subsequence length is less or equal to documents' minimal length."
                      "You should decrease it")
                sys.exit(2)


    def string_kernel(self, X1, X2):
        """
        String Kernel computation
        :param X1: list of documents (m rows, 1 column); each row is a single document (string)
        :type X1: list
        :param X2: list of documents (m rows, 1 column); each row is a single document (string)
        :type X2: list
        :return: Gram matrix for the given parameters
        """
        len_X1 = len(X1)
        len_X2 = len(X2)
        # numpy array of Gram matrix
        gram_matrix = np.zeros((len_X1, len_X2), dtype=np.float32)
        sim_docs_kernel_value = {}
        #when lists of documents are identical
        if X1 == X2:
        #store K(s,s) values in dictionary to avoid recalculations
            for i in range(len_X1):
                sim_docs_kernel_value[i] = self._K(self.subseq_length, X1[i], X1[i])
        #calculate Gram matrix
            for i in range(len_X1):
                for j in range(i, len_X2):
                    gram_matrix[i, j] = self._gram_matrix_element(X1[i], X2[j], sim_docs_kernel_value[i],
                                                                 sim_docs_kernel_value[j])
        #using symmetry
                    gram_matrix[j, i] = gram_matrix[i, j]
        #when lists of documents are not identical but of the same length
        elif len_X1 == len_X2:
            sim_docs_kernel_value[1] = {}
            sim_docs_kernel_value[2] = {}
        #store K(s,s) values in dictionary to avoid recalculations
            for i in range(len_X1):
                sim_docs_kernel_value[1][i] = self._K(self.subseq_length, X1[i], X1[i])
            for i in range(len_X2):
                sim_docs_kernel_value[2][i] = self._K(self.subseq_length, X2[i], X2[i])
        #calculate Gram matrix
            for i in range(len_X1):
                for j in range(i, len_X2):
                    gram_matrix[i, j] = self._gram_matrix_element(X1[i], X2[j], sim_docs_kernel_value[1][i],
                                                                 sim_docs_kernel_value[2][j])
        #using symmetry
                    gram_matrix[j, i] = gram_matrix[i, j]
        #when lists of documents are neither identical nor of the same length
        else:
            sim_docs_kernel_value[1] = {}
            sim_docs_kernel_value[2] = {}
            min_dimens = min(len_X1, len_X2)
        #store K(s,s) values in dictionary to avoid recalculations
            for i in range(len_X1):
                sim_docs_kernel_value[1][i] = self._K(self.subseq_length, X1[i], X1[i])
            for i in range(len_X2):
                sim_docs_kernel_value[2][i] = self._K(self.subseq_length, X2[i], X2[i])
        #calculate Gram matrix for square part of rectangle matrix
            for i in range(min_dimens):
                for j in range(i, min_dimens):
                    gram_matrix[i, j] = self._gram_matrix_element(X1[i], X2[j], sim_docs_kernel_value[1][i],
                                                                 sim_docs_kernel_value[2][j])
                    #using symmetry
                    gram_matrix[j, i] = gram_matrix[i, j]

        #if more rows than columns
            if len_X1 > len_X2:
                for i in range(min_dimens, len_X1):
                    for j in range(len_X2):
                        gram_matrix[i, j] = self._gram_matrix_element(X1[i], X2[j], sim_docs_kernel_value[1][i],
                                                                     sim_docs_kernel_value[2][j])
        #if more columns than rows
            else:
                for i in range(len_X1):
                    for j in range(min_dimens, len_X2):
                        gram_matrix[i, j] = self._gram_matrix_element(X1[i], X2[j], sim_docs_kernel_value[1][i],
                                                                     sim_docs_kernel_value[2][j])
        print(sim_docs_kernel_value)
        return gram_matrix