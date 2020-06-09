import tensorflow as tf
import numpy as np
import editdistance
from vocabulary import Vocabulary
from config import *

v = Vocabulary()


def loss_function(y_true, y_pred):
    indices = tf.argmax(y_true, axis=-1)
    # mask filter padding word
    mask = tf.logical_not(tf.equal(indices, 0))
    mask = tf.cast(mask, dtype=tf.float32)

    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    loss = loss * mask

    return tf.reduce_mean(loss)


def display_validate(generator_valid, model):
    step_per_epoch_validate = len(generator_valid.examples) // BATCH_SIZE

    results = []
    labels = []
    for i in range(step_per_epoch_validate):
        imgs, target = next(generator_valid.examples_generator())
        hidden = tf.zeros((BATCH_SIZE, decode_units))
        word_one_hot = np.zeros((BATCH_SIZE, vocab_size))
        word_one_hot[0][1] = 1.

        for j in range(max_txt_length):
            predict, hidden = model(word_one_hot, hidden, imgs)
            predict_id = tf.argmax(predict, axis=-1)

            word_one_hot = np.zeros((BATCH_SIZE, vocab_size))
            word_one_hot[np.arange(BATCH_SIZE), predict_id.numpy()] = 1.

            results.append(predict)

        results = np.transpose(results, axes=(1, 0, 2))
        for j in range(BATCH_SIZE):
            print('label: {} predict: {}'.format(v.one_hot_decode(target[j]), v.one_hot_decode(results[j])))
        break

# def show_edit_distance(self, num):
#     num_left = num
#     mean_norm_ed = 0.0
#     mean_ed = 0.0
#     while num_left > 0:
#         word_batch = next(self.val_generator.generator())[0]
#         num_proc = min(word_batch['the_inputs'].shape[0], num_left)
#         # predict
#         inputs = word_batch['the_inputs'][0:num_proc]
#         pred = self.y_func([inputs])[0]
#         decoded_res = decode_batch(pred)
#         # label
#         labels = word_batch['the_labels'][:num_proc].astype(np.int32)
#         labels = [labels_to_text(label) for label in labels]
#
#         for j in range(num_proc):
#             edit_dist = editdistance.eval(decoded_res[j], labels[j])
#             mean_ed += float(edit_dist)
#             mean_norm_ed += float(edit_dist) / len(labels[j])
#
#         num_left -= num_proc
#     mean_norm_ed = mean_norm_ed / num
#     mean_ed = mean_ed / num
#     print('\nOut of %d samples:  Mean edit distance:'
