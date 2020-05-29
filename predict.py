from model import Model
from image_util import ImageUtil
from vocabulary import Vocabulary
import numpy as np
import tensorflow as tf

EPOCHS = 2
BATCH_SIZE = 2
LEARNING_RATE = 0.001
embedding_dim = 256
vocab_size = 43 + 3
max_txt_length = 30
decode_units = 256
image_height = 299
image_width = 299

v = Vocabulary()
model = Model(decode_units=decode_units, vocab_size=46, training=True)
model.load_weights('model_epoch0.tf')

image_util = ImageUtil(image_height=image_height, image_width=image_width)
img_tensor = image_util.load('train/00:43 GER 0 - 0 NIR.jpg')
img_tensor = np.expand_dims(img_tensor, 0)

result = ''
hidden = tf.zeros((1, decode_units))
word_one_hot = np.zeros((1, vocab_size))
word_one_hot[0][1] = 1.
for i in range(max_txt_length):
    predict, hidden = model(word_one_hot, hidden, img_tensor)
    predict_id = tf.argmax(predict, axis=-1)

    word_one_hot[0][predict_id.numpy()] = 1.
    word_one_hot = np.zeros((1, vocab_size))

    result += v.one_hot_decode(predict)

print(result)
