import librosa
from keras.utils import np_utils
from scipy.io import wavfile
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import SpeechModels as SM
import DataOrganization as DO

# All classes of the dataset
WORDS = ['backward', 'bed', 'bird', 'cat', 'dog',
         'down', 'eight', 'five', 'follow', 'forward', 'four', 'go',
         'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
         'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up',
         'visual', 'wow', 'yes', 'zero', '_back_noise_']

# Labels for Validation
testWORDS = ['backward', 'bed', 'bird', 'cat', 'dog',
             'down', 'eight', 'five', 'follow', 'forward', 'four', 'go',
             'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
             'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up',
             'visual', 'wow', 'yes', 'zero']

# Mini & Midi -dataset to make test quickly
miniWORDS = ['backward', 'eight', 'learn', 'seven', 'sheila', 'stop']
NUMBERS = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']


def readTestValidation(fileName):
    item_list = []
    with open(fileName) as fp:
        line = fp.readline()
        while line:
            item_list.append(line.strip())
            line = fp.readline()

    return item_list


def rr(waves):
    # The code takes the datapath
    train_audio_path = '/Users/ufukbarankarakaya94/Desktop/WORKSPACE/HDA/SpeechRecognition/dataset'

    duration_of_recordings = []

    for wav in waves:
        sample_rate, samples = wavfile.read(train_audio_path + '/' + wav)
        duration_of_recordings.append(float(len(samples) / sample_rate))

    all_wave = []
    all_label = []

    for wav in waves:
        temp = wav.split('/')
        samples, sample_rate = librosa.load(train_audio_path + '/' + wav, sr=16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if len(samples) == 8000:
            all_wave.append(samples)
            all_label.append(temp[0])

    le = LabelEncoder()
    y = le.fit_transform(all_label)

    y = np_utils.to_categorical(y, num_classes=36)
    all_wave = np.array(all_wave).reshape(-1, 8000, 1)
    testX = np.array(all_wave)
    testY = np.array(y)

    return testX, testY


labs = {WORDS[i]: i + 1 for i in range(0, len(WORDS))}

inputs, labels, x_train, x_validation, y_train, y_validation = DO.arrange()
# Implementation of the Models
itemList1 = readTestValidation('testing_list.txt')
test_x, test_y = rr(itemList1)

'''itemList2 = readTestValidation('validation_list.txt')
val_x, val_y = rr(itemList2)'''
# Convolutional Neural Network - CNN

# SM.CNNSpeechModel(inputs, labels, x_train, y_train, x_validation, y_validation)

# Recurrent Neural Network - RNN
SM.RNNSpeechModel(x_train, y_train, test_x, test_y, 36, samplingrate=8000, inputLength=8000)
# Evaluation of the CNN Model
'''CNN_model = keras.models.load_model('best_model.hdf5')
results = CNN_model.evaluate(test_x, test_y, batch_size=64)
results = CNN_model.evaluate(val_x, val_y, batch_size=64)'''

