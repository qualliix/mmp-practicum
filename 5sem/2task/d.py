from typing import Iterable, Iterator


class linearize:
    def __init__(self, list):
        self.list = list

    def __iter__(self):
        self.cur = None
        self.stack = [iter(self.list)]
        return self

    def __next__(self):
        while True:
            if not self.stack:
                raise StopIteration
            self.cur = self.stack.pop()
            while True:
                if isinstance(self.cur, Iterable) and (not isinstance(self.cur, str) or len(self.cur) > 1):
                    try:
                        if not isinstance(self.cur, Iterator):
                            self.cur = iter(self.cur)
                        temp = next(self.cur)
                        self.stack.append(self.cur)
                        self.cur = temp
                    except StopIteration:
                        break
                else:
                    return self.cur
# print(*list(linearize([
#     (4,3), "mmp", [8, [15, 1], [[6]], [2, [3]]], range(4, 2, -1)
# ])))
