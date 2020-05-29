import numpy as np


class Vocabulary:
    pad = '<PAD>'  # padding
    start = '<START>'  # start of sentence
    eos = '<END>'  # end of sentence

    def __init__(self, letters=" QWERTYUIOPASDFGHJKLZXCVBNM'-:1234567890+()", max_txt_length=30):
        self.vocabulary = ['<PAD>', '<START>', '<END>'] + list(letters)
        self.max_txt_length = max_txt_length
        self.vocab_size = len(self.vocabulary)
        self.character_index = dict([(char, i) for i, char in enumerate(self.vocabulary)])
        self.character_reverse_index = dict((i, char) for char, i in self.character_index.items())

    def one_hot_encode(self, text):
        encoding = np.zeros((self.max_txt_length, self.vocab_size), dtype=np.float32)
        text = ['<START>'] + list(text) + ['<END>']
        text += [self.pad] * (self.max_txt_length - len(text))
        for i, char in enumerate(text):
            encoding[i, self.character_index[char]] = 1

        return encoding

    def one_hot_decode(self, one_hot):
        text = ''
        for index in np.argmax(one_hot, axis=-1):
            sample = self.character_reverse_index[index]

            if sample == self.pad or sample == self.eos or len(text) > self.max_txt_length:
                break
            text += sample

        return text

# v = Vocabulary()
# print(v.vocabulary)
# a = v.one_hot_encode('NGHIA')
# print(a[0])