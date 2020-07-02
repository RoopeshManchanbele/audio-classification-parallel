import code
import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as mfccplt
import matplotlib.pyplot as melplt
import matplotlib.pyplot as chromaplt
import matplotlib.pyplot as contrastplt
import matplotlib.pyplot as tonnetzplt
from matplotlib.pyplot import specgram
import soundfile as sf
import sounddevice as sd
import queue
from os import path
import threading
import time

labelsData = []
featuresData = []
threadLock = threading.Lock()
threads = []


def extract_feature(file_name=None):
    if file_name:
        print('Extracting', file_name)
        X, sample_rate = sf.read(file_name, dtype='float32')
    else:
        device_info = sd.query_devices(None, 'input')
        sample_rate = int(device_info['default_samplerate'])
        q = queue.Queue()

        def callback(i, f, t, s):
            q.put(i.copy())

        data = []
        with sd.InputStream(samplerate=sample_rate, callback=callback):
            while True:
                if len(data) < 100000:
                    data.extend(q.get())
                else:
                    break
        X = np.array(data)

    if X.ndim > 1: X = X[:, 0]
    X = X.T

    # short term fourier transform
    stft = np.abs(librosa.stft(X))

    # mfcc (mel-frequency cepstrum)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


# Overiding thread class
class fileParserThread(threading.Thread):
    def __init__(self, features, file_ext, label, labels, parent_dir, sub_dirs):
        threading.Thread.__init__(self)
        self.features = features
        self.file_ext = file_ext
        self.label = label
        self.labels = labels
        self.parent_dir = parent_dir
        self.sub_dirs = sub_dirs

    # uncomment the Start and End orint statements to the see the number of threads

    def run(self):
        # print("Starting", self.name)
        self.features, self.labels = file_parser(self.features, self.file_ext, self.label, self.labels, self.parent_dir,
                                                 self.sub_dirs)
        for feature in self.features:
            featuresData.append(feature)

        for label in self.labels:
            labelsData.append(label)

        # save the features and labels to the file in run method only
        np.save('feat.npy', featuresData)
        np.save('label.npy', labelsData)
        # print("Exiting", self.name)


def parse_audio_files(parent_dir, file_ext='*.ogg'):
    global mfccs, chroma, mel, contrast, tonnetz
    sub_dirs = os.listdir(parent_dir)
    sub_dirs.sort()
    features, labels = np.empty((0, 193)), np.empty(0)

    # craeating the threads for each
    for label, sub_dir in enumerate(sub_dirs):
        if os.path.isdir(os.path.join(parent_dir, sub_dir)):
            #  creating threads for file parser method

            audioThread = fileParserThread(features, file_ext, label, labels, parent_dir, sub_dir)
            audioThread.start()
            # adding all threads to thread array
            threads.append(audioThread)
            # starting each threads

            # join the threads for synchronous execution
            # comment this for-loop to see the asynchronous execution
            for t in threads:
                t.join()
            # another place where you can see the list of threads
            # print(t)


# Extracted file parsing loop for multithreading
def file_parser(features, file_ext, label, labels, parent_dir, sub_dir):
    global mfccs, chroma, mel, contrast, tonnetz

    for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
        try:
            # Usage of locks to avoid memory access violations threadLock.acquire() and threadLock.release()
            threadLock.acquire()
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            mfccplt.plot(mfccs)
            chromaplt.plot(chroma)
            melplt.plot(mel)
            contrastplt.plot(contrast)
            tonnetzplt.plot(tonnetz)
            threadLock.release()
        except Exception as e:
            print("[Error] extract feature error in %s. %s" % (fn, e))
            continue
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        features = np.vstack([features, ext_features])
        # labels = np.append(labels, fn.split('/')[1])
        labels = np.append(labels, label)

    print("extract %s features done" % sub_dir)
    return np.array(features), np.array(labels, dtype=np.int)


def parse_predict_files(parent_dir, file_ext='*.ogg'):
    features = np.empty((0, 193))
    filenames = []
    for fn in glob.glob(os.path.join(parent_dir, file_ext)):
        mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        features = np.vstack([features, ext_features])
        filenames.append(fn)
        print("extract %s features done" % fn)
    return np.array(features), np.array(filenames)


if __name__ == '__main__':
    startTime = time.time()
    parse_audio_files('data')

    # Predict new
    features, filenames = parse_predict_files('predict')
    np.save('predict_feat.npy', features)
    np.save('predict_filenames.npy', filenames)
    endTime = time.time()
    totalTime = (endTime - startTime) / 60
    print("Execution time =", totalTime)
