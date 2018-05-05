from glob import glob
from json import dump
import numpy as np
import tensorflow as tf


def get_image(fname):
    loaded_ = load_img(fname, target_size=(256, 256), grayscale=True)
    return img_to_array(loaded_)


def read_all_html(start_range, nfiles):
    all_html = glob("data/html_train/*html")
    all_html.sort()
    all_html = all_html[start_range:start_range + nfiles]
    text = []
    for fname in all_html:
        with open(fname, "r") as fi:
            text_ = "<START> " + fi.read() + " <END>"
            text_ = ' '.join(text_.split())
            text_ = text_.replace(',', ' ,')
            text.append(text_)
    return text


def get_tokens():
    from string import printable
    tokens = [_ for _ in printable]
    tokens += ["<p>", "</p>",
               "<h1>", "</h1>",
               "<h2>", "</h2>",
               "<h3>", "</h3>",
               "<ul>", "</ul>",
               "<li>", "</li>",
               "<html>", "</html>",
               "<body>", "</body>",
               "<START>", "<END>"]
    return dict(zip(tokens, range(1, len(tokens) + 1)))


def text_to_token(text, tokens):
    return [tokens[_] for _ in text]


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
            # Cap the input sentence to 60 tokens and add it
            x.append(in_seq[-60:])
            y.append(out_seq)
    return np.array(x), np.array(y), np.array(image_data)


def batch_generator(sequences, max_seq, vocab_size_, jpeg_,
                    bsize):
    i = 0
    while True:
        if i >= len(jpeg_):
            i = 0
        images = [get_image(_) for _ in jpeg_[i:i + bsize]]
        x, y, image_data = preprocess_data(sequences[i:i + bsize], images,
                                           max_seq, vocab_size_)
        i += bsize
        yield [image_data, x], y


def create_model():
    image_model = Sequential()
    image_model.add(Conv2D(16, (3, 3), padding='valid', activation='relu',
                           input_shape=(256, 256, 1,)))
    image_model.add(MaxPooling2D(pool_size=(2, 2)))

    image_model.add(
        Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    image_model.add(MaxPooling2D(pool_size=(2, 2)))

    image_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    image_model.add(MaxPooling2D(pool_size=(2, 2)))

    image_model.add(
        Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    image_model.add(MaxPooling2D(pool_size=(2, 2)))

    image_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    image_model.add(MaxPooling2D(pool_size=(2, 2)))

    image_model.add(
        Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))

    image_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

    image_model.add(Flatten())
    image_model.add(Dense(1024, activation='relu'))
    image_model.add(Dropout(0.3))
    image_model.add(Dense(1024, activation='relu'))
    #     image_model.add(Dropout(0.3))

    image_model.add(RepeatVector(60))

    visual_input = Input(shape=(256, 256, 1,))
    encoded_image = image_model(visual_input)

    language_input = Input(shape=(60,))
    language_model = Embedding(vocab_size, 50, input_length=60,
                               mask_zero=True)(
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
    model_ = Model(inputs=[visual_input, language_input], outputs=decoder)
    optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
    model_.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model_


if __name__ == "__main__":
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    # config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # sess = tf.Session(config = config)
    set_session(tf.Session(config=config))
    # from keras.utils.training_utils import multi_gpu_model
    # from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import Model, Sequential, load_model
    from keras.utils import to_categorical
    from keras.layers.convolutional import Conv2D, MaxPooling2D
    from keras.optimizers import RMSprop
    # from keras.callbacks import ModelCheckpoint
    from keras.layers import Embedding, RepeatVector, LSTM, \
        concatenate, Input, Dense, Flatten, Dropout
    from keras.preprocessing.image import img_to_array, load_img

    # Params
    start = 0
    n_files = 510
    initial_epoch = 0
    epochs_to_run = 300
    filename = None

    # Data setup
    html_ = read_all_html(start, n_files)
    text_token_mapping = get_tokens()
    train_sequences = [text_to_token(_, text_token_mapping) for _ in html_]
    max_sequence = max(len(s) for s in train_sequences)
    vocab_size = len(text_token_mapping) + 1

    jpeg_files = glob("data/jpeg/*.jpeg")
    jpeg_files.sort()
    jpeg_files = jpeg_files[start:start + n_files]

    n_training = 2 * len(train_sequences) // 3
    n_val = len(train_sequences) - n_training

    train_sequences_train = train_sequences[0:n_training]
    jpeg_files_train = jpeg_files[0:n_training]

    train_sequences_val = train_sequences[n_training:]
    jpeg_files_val = jpeg_files[n_training:]

    if filename is None:
        model = create_model()
    else:
        model = load_model(filename)

    # model = multi_gpu_model(model, gpus=2)
    batch_size = 6
    h = model.fit_generator(
        batch_generator(train_sequences_train, max_sequence, vocab_size,
                        jpeg_files_train, batch_size),
        steps_per_epoch=n_training // batch_size,
        epochs=epochs_to_run,
        verbose=1,
        max_queue_size=10,
        initial_epoch=initial_epoch,
        validation_data=batch_generator(train_sequences_val, max_sequence,
                                        vocab_size, jpeg_files_val, batch_size),
        validation_steps=n_val // batch_size
    )

    # loss = model.evaluate_generator(
    #     batch_generator(train_sequences_val, max_sequence, vocab_size,
    #                     jpeg_files_val, batch_size),
    #     steps=n_val // batch_size,
    #     max_queue_size=10
    # )

    last_loss = round(h.history['loss'][-1], 2)
    last_val_loss = round(h.history['val_loss'][-1], 2)

    filename = "start_{file_start}_end_{file_end}_epochs_{epoch_start}_to_" \
               "{epoch_end}_loss_{loss}_valLoss_{val_loss}".format(
                loss=last_loss,
                val_loss=last_val_loss,
                file_start=start,
                file_end=start + n_files,
                epoch_start=initial_epoch,
                epoch_end=initial_epoch + epochs_to_run)

    with open(filename + ".json", "w") as f:
        dump(h.history, f)

    model.save("model_outputs/" + filename + ".h5")
