def find_path_sums(tree, sum=0):
    right = tree[2]
    left = tree[1]
    if (left):
        find_path_sums(left, sum+tree[0])
    if (right):
        find_path_sums(right, sum+tree[0])
    if (right is None and left is None):
        print(sum + tree[0])
    return

