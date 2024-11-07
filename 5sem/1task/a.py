def find_word_in_circle(circle, word):
    le = len(circle)
    word_len = len(word)
    if (le == 0):
        return -1
    if (le < word_len):
        dir = 1
        if (circle in word):
            t = circle
        elif (circle[::-1] in word):
            t = circle[::-1]
            dir = -1
        else:
            return -1
        t = t * ((word_len // le) + 1)
        if (word in t):
            return t.find(word) % le, dir
        else:
            return -1
    s = circle + circle
    reverse = word[::-1]
    if (word in s):
        return s.find(word) % le, 1
    if (reverse in s):
        return ((s.find(reverse) + word_len) % le) - 1, -1
    else:
        return -1
