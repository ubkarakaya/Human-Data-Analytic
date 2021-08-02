import os
import librosa  # for audio processing
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.utils import np_utils


def arrange():
    # The code takes the datapath
    train_audio_path = '/Users/ufukbarankarakaya94/Desktop/WORKSPACE/HDA/SpeechRecognition/dataSet'

    labels = os.listdir(train_audio_path)

    labels.remove('.DS_Store')
    # find count of each label and plot bar graph
    no_of_recordings = []
    for label in labels:
        waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
        no_of_recordings.append(len(waves))

    # drawGraph(labels, no_of_recordings)
    duration_of_recordings = []
    for label in labels:
        waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
        for wav in waves:
            sample_rate, samples = wavfile.read(train_audio_path + '/' + label + '/' + wav)
            duration_of_recordings.append(float(len(samples) / sample_rate))

    all_wave = []
    all_label = []
    for label in labels:
        print(label)
        waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
        for wav in waves:
            samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr=16000)
            samples = librosa.resample(samples, sample_rate, 8000)
            if len(samples) == 8000:
                all_wave.append(samples)
                all_label.append(label)

    le = LabelEncoder()
    y = le.fit_transform(all_label)

    y = np_utils.to_categorical(y, num_classes=len(labels))
    all_wave = np.array(all_wave).reshape(-1, 8000, 1)

    # Splitting of data as train and validation set to train and test our model
    x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave), np.array(y), stratify=y, test_size=0.2, random_state=777, shuffle=True)

    K.clear_session()
    inputs = Input(shape=(8000, 1))

    return inputs, labels, x_tr, x_val, y_tr, y_val


def drawGraph(labels, no_of_recordings):
    # plot
    plt.figure(figsize=(30, 5))
    index = np.arange(len(labels))
    plt.bar(index, no_of_recordings)
    plt.xlabel('Commands', fontsize=12)
    plt.ylabel('No of recordings', fontsize=12)
    plt.xticks(index, labels, fontsize=15, rotation=60)
    plt.title('No. of recordings for each command')
    plt.show()
