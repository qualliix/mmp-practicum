def find_path_sums(tree):
    stack = []
    node = tree
    sum = 0
    while True:
        left = node[1]
        right = node[2]
        sum += node[0]
        if (left):
            if (right):
                stack.append((right, sum))
            node = left
        elif (right):
            node = right
        else:
            print(sum)
            if not stack:
                break
            t = stack.pop()
            node = t[0]
            sum = t[1]
    return
