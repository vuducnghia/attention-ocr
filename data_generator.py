from image_util import ImageUtil
from vocabulary import Vocabulary
import os
import random
import numpy as np

letters = " QWERTYUIOPASDFGHJKLZXCVBNM'-:1234567890+()"


class Generator:
    def __init__(self, folder_image, folder_label, batch_size, image_height=32, image_width=320, max_txt_length=30):
        self.folder_image = folder_image
        self.folder_label = folder_label
        self.batch_size = batch_size
        self.max_txt_length = max_txt_length
        self.examples = []
        self.cur_index = 0
        self.load_data()
        self.image_util = ImageUtil(image_height=image_height, image_width=image_width)
        self.vocab = Vocabulary(letters, self.max_txt_length)

    def load_data(self):
        with open(self.folder_label, 'r') as f:
            for line in f.readlines():
                if ';' in line:
                    image_file, txt = line.split(sep=';', maxsplit=1)
                    image_file = os.path.abspath(os.path.join(self.folder_image, image_file))
                    txt = txt.strip()
                    if os.path.isfile(image_file):
                        self.examples.append((txt, image_file))


    def examples_generator(self):
        random.shuffle(self.examples)
        while True:
            images, target, encode_hidden = [], [], []
            for i in range(self.batch_size):
                self.cur_index += 1
                if self.cur_index >= len(self.examples):
                    self.cur_index = 0


                txt, img_path = self.examples[self.cur_index]
                # print(txt, img_path)
                images.append(self.image_util.load(img_path))
                target.append(self.vocab.one_hot_encode(txt))
                # print(self.vocab.text_to_labels(txt))
                # print(self.vocab.labels_to_text(target[0]))
            yield np.array(images), np.array(target)
