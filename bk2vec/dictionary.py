from __future__ import absolute_import
from __future__ import print_function

class Dictionary():
    def __init__(self):
        self.unknown = 'UNK'
        self.dict = {self.unknown: 0}
        self.rev_dict = {0: self.unknown}

    def __len__(self):
        return len(self.dict)

    def add_word(self, word):
        self.put_word(len(self.dict), word)

    def put_word(self, index, word):
        self.dict[word] = index
        self.rev_dict[index] = word

    def has_word(self, word):
        return word in self.dict

    def __getitem__(self, item):
        return self.lookup(item)

    def lookup(self, word):
        if not self.has_word(word):
            word = self.unknown
        return self.dict[word]