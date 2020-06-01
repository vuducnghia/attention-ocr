import tensorflow as tf
from model import Model
import time
from data_generator import Generator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

EPOCHS = 2
BATCH_SIZE = 2
LEARNING_RATE = 0.001
embedding_dim = 256
vocab_size = 43 + 3
max_txt_length = 30
decode_units = 256
image_height = 299
image_width = 299
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)


def loss_function(y_true, y_pred):
    indices = tf.argmax(y_true, axis=-1)
    # mask filter padding word
    mask = tf.logical_not(tf.equal(indices, 0))
    mask = tf.cast(mask, dtype=tf.float32)

    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    loss = loss * mask

    return tf.reduce_mean(loss)


model = Model(decode_units=decode_units, vocab_size=46, training=True)


@tf.function
def train_step(images, word_target):  # word_target shape(bs, max_txt_length, vocab_size)

    loss = 0

    hidden = tf.zeros((BATCH_SIZE, decode_units))
    word_one_hot = word_target[:, 0, :]  # corresponding the word 'START'
    with tf.GradientTape() as tape:
        # Teacher forcing - feeding the target as the next input
        for i in range(1, word_target.shape[1]):
            y_pred, hidden = model(word_one_hot, hidden, images)
            word_one_hot = word_target[:, i, :]

            loss += loss_function(word_target[:, i, :], y_pred)

    batch_loss = loss / int(word_target.shape[1])
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


if __name__ == '__main__':
    generator_training = Generator(folder_image='train',
                                   folder_label='train.txt',
                                   batch_size=BATCH_SIZE,
                                   image_height=image_height,
                                   image_width=image_width,
                                   max_txt_length=max_txt_length)

    # if args['finetune']:
    #     model.load_weights('model.h5')

    print(len(generator_training.examples))
    # print(len(generator_valid.examples))

    step_per_epoch_training = len(generator_training.examples) // BATCH_SIZE
    # step_per_epoch_validate = len(generator_valid.examples) // BATCH_SIZE

    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0

        for i in range(step_per_epoch_training):
            imgs, target = next(generator_training.examples_generator())
            total_loss += train_step(imgs, target)

        if epoch % 10 == 0:
            model.save_weights(f'model_epoch{epoch}.h5')

        print('Epoch {}/{} Loss {:.6f}'.format(epoch + 1, EPOCHS, total_loss / step_per_epoch_training))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
