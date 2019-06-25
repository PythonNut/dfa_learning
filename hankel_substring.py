import numpy as np
import copy
from numpy.linalg import matrix_rank
from functools import reduce
from itertools import permutations 


# slightly modified from WKPlus' post on stack overflow
def ordered_combination(s):
    '''returns list of all possible splicings of string s '''
    result = []
    for i in range(len(s)):
        for j in range(i+1, len(s)+1):
            result.append(s[i:j])
    return result


def generate_basis(s, length_bound):
    ''' returns list of all possible two-partitions of every word, unique prefixes for prefix indices, 
    unique suffixes for suffix indices, and all possible substrings '''

    # partitions into prefixes and suffixes
    partitions_list = []
    all_substrings = []

    for w in s:
        # only consider partitions into prefixes and suffixes for hankell matrix indices
        partitions_list += list(filter(lambda x: len(x) <= 2,all_partitions(w)))

        # we want all possible partitions of all possible strings for the count dictionary later
        all_substrings += all_partitions(w)

    prefixes = list(map(lambda x: x[0], partitions_list))

    # all suffixes also includes case where epsilon is prefix and w is suffix
    epsilon_prefix = list(map(lambda x: x[0], list(filter(lambda x: len(x) == 1, partitions_list))))
    suffixes = list(map(lambda x: x[1], list(filter(lambda x: len(x)> 1, partitions_list)))) + epsilon_prefix
    
    # we want unique prefixes and suffixes to determine our hankell matrix indices
    unique_prefixes = sorted(list(filter(lambda x: len(x)<= length_bound, list(set(prefixes)))))
    unique_suffixes = sorted(list(filter(lambda x: len(x)<= length_bound, list(set(suffixes)))))
    return  (partitions_list,unique_prefixes, unique_suffixes, all_substrings)


def substring_count(s, all_substrings):
    '''returns a dictionary of the counts of each substring'''
    count_dict = {}
    num_strings = len(s)
    ps = list(set(sum(all_substrings, [])))
    for p in ps:
        count_dict[p] = 0
    for w in s:
        for x in ps:
            # how many times a substring occurs in a word in the set of all training strings
            count = ordered_combination(w).count(x)
            # add count fraction
            count_dict[x] += count/float(num_strings)
    return count_dict

# from johnLate's post on stack overflow 
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

def hankel(s,length_bound):
    '''returns prefixes, suffixes, and hankel string statistics matrix'''
    prefix_index = {}
    suffix_index = {}
    EPSILON_INDEX = 1
    partitions_list,unique_prefixes, unique_suffixes, all_substrings = generate_basis(s,length_bound)
    count_dict = substring_count(s,all_substrings)

    prefix_indices_length = len(unique_prefixes)
    suffix_indices_length = len(unique_suffixes)

    hankel_matrix = np.zeros((prefix_indices_length+EPSILON_INDEX,suffix_indices_length+EPSILON_INDEX))
    # add the value i+EPSILON_INDEX to the key unique_prefixes[index] to account for epsilon, empty string. 
    # we want easy access to where we should put values in our hankel matrix; hence, 
    # we construct a dictionary with the word as the key and the index as the value whilst accounting for epsilon at index 0
    for index in range(prefix_indices_length):
        prefix_index[unique_prefixes[index]] = index+EPSILON_INDEX
    for index in range(suffix_indices_length):
        suffix_index[unique_suffixes[index]] = index+EPSILON_INDEX
    
 
    for partition in partitions_list:
        # we only look at substrings of length less than or equal to 2

        # store as variables to avoid unnecessary calculations/retrievals
        prefix = partition[0]
        prefix_length = len(prefix)
        partition_length = len(partition)

        # this is because we want prefixes that are indexes in our matrix
        if prefix_length <= length_bound:
            if (prefix in prefix_index.keys()) and (prefix in suffix_index.keys()):
                hankel_matrix[prefix_index[prefix]][0] = count_dict[prefix]
                hankel_matrix[0][suffix_index[prefix]] = count_dict[prefix]

        # now considering partitions that are length two (split into non-empty prefix and suffix)
        if partition_length==2 and prefix_length <= length_bound and len(partition[1])<=length_bound:
            # store as variable to avoid unnecessary calculations/retrievals
            suffix = partition[1]
            if (prefix in prefix_index.keys()) and (suffix in suffix_index.keys()):
                hankel_matrix[prefix_index[prefix]][suffix_index[suffix]] = count_dict[prefix + suffix]
                hankel_matrix[0][suffix_index[suffix]] = count_dict[suffix]
    
    return (["epsilon"] + unique_prefixes,
   ["epsilon"] + unique_suffixes,
    hankel_matrix)


    return partitions_list



print(hankel(["aa","b","bab","a","bbab","abb","babba","abbb","ab","a","aabba","baa","abbab","baba","bb","a"],3))