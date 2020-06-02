from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, GRU
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

InceptionResNetV2 = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False,
                                                            input_tensor=Input(shape=(299, 299, 3)))
# InceptionResNetV2.summary()
base_model = Model(inputs=InceptionResNetV2.get_layer('input_1').input,
                   outputs=InceptionResNetV2.get_layer('mixed_6a').output)
base_model.summary()  # output (bs, 17, 17, 1088)


class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, hidden_state, features):
        hidden_with_time_axis = tf.expand_dims(hidden_state, 1)
        hidden_with_time_axis = tf.expand_dims(hidden_with_time_axis, 1)

        score = self.V(tf.nn.tanh(self.W1(hidden_with_time_axis) + self.W2(features)))  # (bs, 17, 17 ,1)
        s1 = tf.nn.softmax(score, axis=1)
        s2 = tf.nn.softmax(score, axis=2)
        attention_weight = tf.reduce_sum([s1, s2], 0)

        context_vector = attention_weight * features  # (bs, 17, 17, 1088)
        context_vector = tf.reduce_sum(context_vector, [1, 2])  # (bs, 1088)

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
        # output_predict = tf.argmax(output_predict, axis=-1)  # (bs,)

        return output_predict, current_hidden


class Model(tf.keras.Model):
    def __init__(self, decode_units=256, vocab_size=46, training=True):
        super().__init__()
        self.training = training

        self.decoder = Decoder(decode_units, vocab_size)

    def call(self, word_one_hot, pre_hidden, images):
        features = base_model(images)
        logits, hidden = self.decoder(word_one_hot, pre_hidden, features)

        return logits, hidden

# m = Model()
# # m.build(input_shape=(2, 299, 299, 3))
# # m.summary()
# img = cv2.imread('1.jpg')
# img = cv2.resize(img, (299, 299))
#
# img = np.asarray([img, img])
# print(img.shape)
# m(tf.ones((2, 10)), tf.ones((2, 256)), img)
