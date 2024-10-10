from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys
# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# to play the audio files
from IPython.display import Audio
import keras
from keras.models import load_model
from keras.utils import to_categorical
import np_utils
from keras.callbacks import ModelCheckpoint
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# Initializing flask application
app = Flask(__name__)
cors = CORS(app)

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(y=data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)

def extract_features(data,sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally
    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data,sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    # without augmentation
    res1 = extract_features(data,sample_rate)
    result = np.array(res1)
    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data,sample_rate)
    result = np.vstack((result, res2)) # stacking vertically
    
    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch,sample_rate)
    result = np.vstack((result, res3)) # stacking vertically 
    
    return result

#Load model
ldmd=load_model('models\lstm_model.h5')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def prediction():
    p= request.files['audiofile']
    x, y = [], []

    feature = get_features(p)
    for ele in feature:
        x.append(ele)
        

    # scaling our data with sklearn's Standard scaler
    scaler = StandardScaler()
    xt = scaler.fit_transform(x)
    xt = np.expand_dims(xt, axis=2)
    predtest = ldmd.predict(xt)

    Y=['neutral', 'calm', 'happy','sad','angry','fear','disgust','surprise']
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
    output = encoder.inverse_transform(predtest)
    emotion=output.flatten()
    
    classification = ['','']

    classification[0] = emotion[1]  
    classification[1] = emotion

    return (classification[0])

if __name__ == '__main__':
    app.run()