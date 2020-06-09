from tensorflow.keras.layers import Input, Dense, GRU
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, hidden_state, features):
        hidden_with_time_axis = tf.expand_dims(hidden_state, 1)

        score = self.V(tf.nn.tanh(self.W1(hidden_with_time_axis) + self.W2(features)))  # (bs, -1 ,1)
        attention_weight = tf.nn.softmax(score, axis=1)  # (bs, length, 1)

        context_vector = attention_weight * features  # (bs, lenght, 1088)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (bs, 1088)

        return context_vector, attention_weight


class Decoder(tf.keras.layers.Layer):
    def __init__(self, decode_units, vocab_size):
        super().__init__()
        self.attention = Attention(units=decode_units)  # units :any (eg: 32, 64, ...), maybe not decode_units

        self.RNN = GRU(units=decode_units, return_sequences=True, return_state=True,
                       recurrent_initializer='glorot_uniform')
        self.W1 = Dense(decode_units)
        self.W2 = Dense(decode_units)
        self.W3 = Dense(vocab_size)
        self.W4 = Dense(vocab_size)

    def call(self, word_one_hot, pre_hidden, features):
        context_vector, attention_weight = self.attention(pre_hidden, features)  # context : (bs, 1088)
        x = self.W1(word_one_hot) + self.W2(context_vector)  # (bs, 1, feature_size)
        x = tf.expand_dims(x, axis=1)  # (bs, 1, units)

        output, current_hidden = self.RNN(inputs=x, initial_state=pre_hidden)  # output: (bs, 1, decode_units)
        output = tf.reshape(output, shape=(-1, output.shape[2]))

        output_predict = self.W3(output) + self.W4(context_vector)  # (bs, vocab_size)
        output_predict = tf.nn.softmax(output_predict, axis=-1)

        return output_predict, current_hidden, attention_weight


class Model(tf.keras.Model):
    def __init__(self, decode_units=256, vocab_size=46, image_height=299, image_width=299,
                 finetune=False, visual_attention=False):
        super().__init__()
        InceptionResNetV2 = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False,
                                                                    input_tensor=Input((image_height, image_width, 3)))

        self.base_model = tf.keras.models.Model(inputs=InceptionResNetV2.input,
                                                outputs=InceptionResNetV2.get_layer('mixed_6a').output)
        _, self.h, self.w, _ = self.base_model.layers[-1].output.shape

        self.base_model.summary()  # output (bs, 17, 17, 1088)
        self.visual_attention = visual_attention
        if finetune:
            self.base_model.trainable = True
        else:
            self.base_model.trainable = False

        self.decoder = Decoder(decode_units, vocab_size)

    def call(self, word_one_hot, pre_hidden, images):
        features = self.base_model(images)

        # Adding pixel coordinates to image features
        batch_size, h, w, _ = features.shape
        x, y = tf.meshgrid(tf.range(w), tf.range(h))
        w_loc = tf.one_hot(x, depth=w)
        h_loc = tf.one_hot(y, depth=h)
        loc = tf.concat([h_loc, w_loc], axis=2)
        loc = tf.tile(tf.expand_dims(loc, 0), multiples=[batch_size, 1, 1, 1])
        features = tf.concat([features, loc], axis=3)  # (batch_size, height, width, num_features + coord)

        feature_size = features.shape[3]
        features = tf.reshape(features, shape=(batch_size, -1, feature_size))  # (batch_size, seq_length, features_size)

        logits, hidden, attention_weight = self.decoder(word_one_hot, pre_hidden, features)

        if self.visual_attention:
            return logits, hidden, attention_weight
        else:
            return logits, hidden


# import cv2
# import numpy as np
#
# m = Model(image_height=80, image_width=320)
# img1 = cv2.imread('train/00:43 GER 0 - 0 NIR.jpg')
# img1 = cv2.resize(img1, (320, 80))
# img2 = cv2.imread('train/01:29 MAINZ 0-0 LEIPZIG.jpg')
# img2 = cv2.resize(img2, (320, 80))
#
# img = np.array([img1, img2])
# img = img / 255.0
# woh = np.random.random(size=(2, 46))
# preh = tf.zeros((2, 256))
# a = m(woh, preh, img)

