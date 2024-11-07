def find_max_substring_occurrence(input_string):
    k = len(input_string)
    for t in range(1, (k//2)+1):
        if (input_string[:t]*(k//t) == input_string):
            return k//t
    return 1
