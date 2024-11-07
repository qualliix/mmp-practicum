class WordContextGenerator:
    def __init__(self, words, k):
        self.out = list()
        self.words = words
        self.k = k
        self.len = len(words)

    def __iter__(self):
        for i in range(0, self.len):
            for j in range(max(0, i-self.k), min(i+self.k + 1, self.len)):
                if i != j:
                    yield (self.words[i], self.words[j])
