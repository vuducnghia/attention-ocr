from model import Model
from image_util import ImageUtil
from vocabulary import Vocabulary
from config import *
import numpy as np
import tensorflow as tf
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', default="images/03:43   KCORVDB   3 - 3   ('I.jpg", help='link to image')
args = parser.parse_args()
v = Vocabulary()
model = Model(decode_units=decode_units,
              vocab_size=vocab_size,
              image_height=image_height,
              image_width=image_width,
              finetune=False,
              visual_attention=True)
model.load_weights('model/model_epoch40')
height, width = model.h, model.w

def show_origin_image():
    img = cv2.imread(args.image)
    # img = cv2.resize(img, (width, height))
    cv2.imshow('origin', img)


def visual_attention(result, attention_plot):
    len_result = len(result)
    for i in range(len_result):
        show_origin_image()
        temp_att = np.reshape(attention_plot[i], (height, width))

        cv2.imshow(f'predict word: {result[i]}', temp_att)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    image_util = ImageUtil(image_height=image_height, image_width=image_width)
    img_tensor = image_util.load(args.image)
    img_tensor = np.expand_dims(img_tensor, 0)

    result = ''
    hidden = tf.zeros((1, decode_units))
    word_one_hot = np.zeros((1, vocab_size))
    word_one_hot[0][1] = 1.

    attention_plot = np.zeros((max_txt_length, height * width))

    for i in range(max_txt_length):
        predict, hidden, attention_weights = model(word_one_hot, hidden, img_tensor)
        predict_id = tf.argmax(predict, axis=-1)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
        if predict_id == 0 or predict_id == 2:
            break

        word_one_hot = np.zeros((1, vocab_size))
        word_one_hot[0][predict_id.numpy()] = 1.

        result += v.one_hot_decode(predict)

    print(result)
    attention_plot = attention_plot[:len(result), :]
    visual_attention(result, attention_plot)
