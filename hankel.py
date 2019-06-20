import numpy as np
import copy


def hankel(s):
    '''returns prefixes, suffixes, and hankel matrix'''
    pd = {}
    sd = {}

    partitions_list = []
    for w in s:
        # only consider partitions into prefixes and suffixes
        partitions_list += list(filter(lambda x: len(x) <= 2,all_partitions(w)))

    prefixes = list(map(lambda x: x[0], partitions_list))
    # all suffixes also includes case where epsilon is prefix and w is suffix
    epsilon_prefix = list(map(lambda x: x[0], list(filter(lambda x: len(x) == 1, partitions_list))))
    suffixes = list(map(lambda x: x[1], list(filter(lambda x: len(x)> 1, partitions_list)))) + epsilon_prefix

    # for our row and col labels, we only want to consider the unique prefixes and suffixes
    unique_prefixes = list(set(prefixes))
    unique_suffixes =  list(set(suffixes))

    # we want to add to each element the proportion it shows up in s
    decimal = 1.0/(len(s))
    EPSILON = 1

    hankel_matrix = np.zeros((len(unique_prefixes)+EPSILON,len(unique_suffixes)+EPSILON))

    # add the value i+EPSILON to the key unique_prefixes[i] to account for epsilon, empty string. 
    # we want easy access to where we should put values in our hankel matrix; hence, 
    # the dictionary with the word as the key and the index as the value
    for i in range(len(unique_prefixes)):
        pd[unique_prefixes[i]] = i+EPSILON
    for i in range(len(unique_suffixes)):
        sd[unique_suffixes[i]] = i+EPSILON
    

    # now, we look at all possible partitions and add them to our hankel matrix using the prefix and suffix dictionaries
    for p in partitions_list:
        if len(p) == 1:
            hankel_matrix[pd[p[0]]][0] += decimal
            hankel_matrix[0][sd[p[0]]] += decimal
        else:
            hankel_matrix[pd[p[0]]][sd[p[1]]] += decimal
    return (unique_prefixes,
    unique_suffixes,
    hankel_matrix)

# from stack overflow by johnLate 
def all_partitions(string):
    for cutpoints in range(1 << (len(string)-1)):
        result = []
        lastcut = 0
        for i in range(len(string)-1):
            if (1<<i) & cutpoints != 0:
                result.append(string[lastcut:(i+1)])
                lastcut = i+1
        result.append(string[lastcut:])
        yield result



