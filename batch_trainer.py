from glob import glob


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#sess = tf.Session(config = config)
set_session(tf.Session(config = config))


from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, RepeatVector, LSTM, \
    concatenate, Input,  Dense, Flatten, Dropout
from keras.preprocessing.image import img_to_array, load_img
import numpy as np


def get_image(fname):
    loaded_ = load_img(fname, target_size=(256, 256))
    return img_to_array(loaded_)


def read_all_html():
    all_html = glob("data/html_train/*html")
    all_html.sort()
    text = []
    for filename in all_html:
        with open(filename, "r") as f:
            text_ = "<START> " + f.read() + " <END>"
            text_ = ' '.join(text_.split())
            text_ = text_.replace(',', ' ,')
            text.append(text_)
    return text


def build_vocab(html_text):
    # Initialize the function to create the vocabulary
    tokenizer = Tokenizer(filters='', split=" ", lower=False)
    # Create the vocabulary
    tokenizer.fit_on_texts(html_text)
    # Add +1 to leave space for empty words
    voc_size = len(tokenizer.word_index) + 1
    # Translate each word in text file to the matching vocabulary index
    sequences = tokenizer.texts_to_sequences(html_text)
    # The longest HTML file
    max_seq = max(len(s) for s in sequences)
    max_len = 48
    return sequences, max_seq, max_len, voc_size


def preprocess_data(sequences, features, max_seq, voc_size):
    x, y, image_data = list(), list(), list()
    for img_no, seq in enumerate(sequences):
        for i in range(1, len(seq)):
            # Add the sentence until the current count(i) and add the current
            #  count to the output
            in_seq, out_seq = seq[:i], seq[i]
            # Pad all the input token sentences to max_sequence
            in_seq = pad_sequences([in_seq], maxlen=max_seq)[0]
            # Turn the output into one-hot encoding
            out_seq = to_categorical([out_seq], num_classes=voc_size)[0]
            # Add the corresponding image to the boostrap token file
            image_data.append(features[img_no])
            # Cap the input sentence to 48 tokens and add it
            x.append(in_seq[-48:])
            y.append(out_seq)
    return np.array(x), np.array(y), np.array(image_data)


def batch_generator(sequences, max_seq, vocab_size_, jpeg_,
                    batch_size):
    print ("------")
    i = 0
    while True:
        if i >= len(jpeg_):
            i = 0
        images = [get_image(_) for _ in jpeg_[i:i + batch_size]]
        x, y, image_data = preprocess_data(sequences[i:i + batch_size],
                                           images, max_seq, vocab_size_)
        i += batch_size
        yield [image_data, x], y


if __name__ == "__main__":
    html_ = read_all_html()
    train_sequences, max_sequence, max_length, vocab_size = build_vocab(html_)
    jpeg_files = glob("data/jpeg/*.jpeg")
    jpeg_files.sort()

    n_training = 2 * len(train_sequences) // 3
    n_val = len(train_sequences) - n_training

    train_sequences_train = train_sequences[0:n_training]
    jpeg_files_train = jpeg_files[0:n_training]

    train_sequences_val = train_sequences[n_training:]
    jpeg_files_val = jpeg_files[n_training:]

    # Create the encoder
    image_model = Sequential()
    image_model.add(Conv2D(16, (3, 3), padding='valid', activation='relu',
                           input_shape=(256, 256, 3,)))
    image_model.add(
        Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    image_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    image_model.add(
        Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    image_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    image_model.add(
        Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    image_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

    image_model.add(Flatten())
    image_model.add(Dense(1024, activation='relu'))
    image_model.add(Dropout(0.3))
    image_model.add(Dense(1024, activation='relu'))
    image_model.add(Dropout(0.3))

    image_model.add(RepeatVector(48))

    visual_input = Input(shape=(256, 256, 3,))
    encoded_image = image_model(visual_input)

    language_input = Input(shape=(48,))
    language_model = Embedding(vocab_size, 50, input_length=48, mask_zero=True)(
        language_input)
    language_model = LSTM(128, return_sequences=True)(language_model)
    language_model = LSTM(128, return_sequences=True)(language_model)

    # Create the decoder
    decoder = concatenate([encoded_image, language_model])
    decoder = LSTM(512, return_sequences=True)(decoder)
    decoder = LSTM(512, return_sequences=False)(decoder)
    decoder = Dense(vocab_size, activation='softmax')(decoder)

    # Compile the model
    # with tf.device("/cpu:0"):
    model = Model(inputs=[visual_input, language_input], outputs=decoder)
    optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # model = multi_gpu_model(model, gpus=1)
    batch_size = 1
    model.fit_generator(
        batch_generator(train_sequences_train, max_sequence, vocab_size,
                        jpeg_files_train, batch_size),
        validation_data=batch_generator(train_sequences_val, max_sequence,
                                        vocab_size, jpeg_files_val, batch_size),
        validation_steps=n_val//batch_size,
        steps_per_epoch=n_training//batch_size,
        epochs=5,
        verbose=1,
        max_queue_size=1)
