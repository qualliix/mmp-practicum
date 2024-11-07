from itertools import zip_longest


def check_first_sentence_is_second(s1, s2):
    dict1 = dict()
    dict2 = dict()
    for word1, word2 in zip_longest(s1.split(), s2.split()):
        if (word1 in dict1):
            dict1[word1] += 1
        elif word1 is not None:
            dict1[word1] = 1
        if (word2 in dict2):
            dict2[word2] += 1
        elif word2 is not None:
            dict2[word2] = 1
    for word in dict2.keys():
        if (not (word in dict1.keys())):
            return False
        elif dict1[word] < dict2[word]:
            return False
    return True
