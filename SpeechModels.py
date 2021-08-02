import warnings
warnings.filterwarnings("ignore")
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def CNNSpeechModel(inputs, labels, train_x, train_y, validation_x, validation_y):
    # First Conv1D layer
    conv = Conv1D(8, 13, padding='valid', activation='relu', strides=1)(inputs)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    # Second Conv1D layer
    conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    # Third Conv1D layer
    conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    # Fourth Conv1D layer
    conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    # Flatten layer
    conv = Flatten()(conv)

    # Dense Layer 1
    conv = Dense(256, activation='relu')(conv)
    conv = Dropout(0.3)(conv)

    # Dense Layer 2
    conv = Dense(128, activation='relu')(conv)
    conv = Dropout(0.3)(conv)

    outputs = Dense(len(labels), activation='softmax')(conv)

    model = Model(inputs, outputs)
    model.summary()
    # Initialization of early-stopping and checkpoint mechanisms
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
    mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # Fitting of the model
    result = model.fit(train_x, train_y, epochs=100, callbacks=[es, mc], batch_size=64,
                       validation_data=(validation_x, validation_y))
    model.save('best_model.hdf5')


def RNNSpeechModel(x_train, y_train, x_validation, y_validation, nCategories, samplingrate=16000,
                      inputLength=16000, rnn_func=L.LSTM):
    # simple LSTM
    sr = samplingrate
    iLen = inputLength

    inputs = L.Input((inputLength,), name='input')

    x = L.Reshape((1, -1))(inputs)

    m = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, iLen),
                       padding='same', sr=sr, n_mels=80,
                       fmin=40.0, fmax=sr / 2, power_melgram=1.0,
                       return_decibel_melgram=True, trainable_fb=False,
                       trainable_kernel=False,
                       name='mel_stft')
    m.trainable = False

    x = m(x)

    x = Normalization2D(int_axis=0, name='mel_stft_norm')(x)

    x = L.Permute((2, 1, 3))(x)

    x = L.Conv2D(10, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(1, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)

    x = L.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')(x)

    x = L.Bidirectional(rnn_func(64, return_sequences=True))(x)
    x = L.Bidirectional(rnn_func(64, return_sequences=True))(x)

    xFirst = L.Lambda(lambda q: q[:, -1])(x)
    query = L.Dense(128)(xFirst)

    # dot product attention
    attScores = L.Dot(axes=[1, 2])([query, x])
    attScores = L.Softmax(name='attSoftmax')(attScores)

    # rescale sequence
    attVector = L.Dot(axes=[1, 1])([attScores, x])

    x = L.Dense(64, activation='relu')(attVector)
    x = L.Dense(32)(x)

    output = L.Dense(nCategories, activation='softmax', name='output')(x)

    model = Model(inputs=[inputs], outputs=[output])

    model.compile(optimizer='adam', loss=['categorical_crossentropy'], metrics=['accuracy'])
    earlystopper = EarlyStopping(monitor='accuracy', patience=10, verbose=1, restore_best_weights=True)
    checkpointer = ModelCheckpoint('modelRNN.hdf5', monitor='accuracy', verbose=1, save_best_only=True)

    results = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=60, verbose=1, batch_size=32, callbacks=[earlystopper, checkpointer])
    model.save('modelRNN.hdf5')

