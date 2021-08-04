import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

# keras
import keras
from keras import models, optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, GRU
from keras import Model ,models, layers, optimizers, regularizers
from keras.callbacks import ModelCheckpoint
import keras.backend.tensorflow_backend as K
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

class DetectModels:
    def __init__(self, unit=128, inputshape=(10,78), timesteps=10, dropout=0.3, lr=0.001, batchsize=100, epochs=10, verbose=1):
        self.unit = unit
        self.inputshape = inputshape
        self.timesteps = timesteps
        self.dropout = dropout
        self.lr = lr
        self.batchsize = batchsize
        self.epochs = epochs
        self.verbose = verbose

    def checkFolder(self):
        try:
            if not os.path.exists("../models/"):
                os.makedirs("../models/")
        except OSError:
            print ('Error: Creating directory. ')

    def trainTestSplit(self, train_df):
        train, test = train_test_split(train_df, test_size=0.3)
        return zip(train, test)

    def reshapeTo3D(self):
        # 고칠꺼
        x_train_y0_rest = -(x_train_y0_scaled.shape[0] % timesteps)
        x_valid_y0_rest = -(x_valid_y0_scaled.shape[0] % timesteps)
        x_valid_rest = -(x_valid_scaled.shape[0] % timesteps)
        x_test_rest = -(x_test_scaled.shape[0] % timesteps)

        x_train_y0_scaled = x_train_y0_scaled[:x_train_y0_rest]
        x_valid_y0_scaled = x_valid_y0_scaled
        x_valid_scaled = x_valid_scaled[:x_valid_rest]
        x_test_scaled = x_test_scaled[:x_test_rest]

        valid_label_rest = valid_label[:x_valid_rest]
        test_label_rest = test_label[:x_test_rest]

        # reshape input to be 3D [samples, timesteps, features]
        x_train_y0_scaled_reshape = x_train_y0_scaled.reshape((int(x_train_y0_scaled.shape[0]/timesteps), timesteps, x_train_y0_scaled.shape[1])) # 정상 데이터 셋
        x_valid_y0_scaled_reshape = x_valid_y0_scaled.reshape((int(x_valid_y0_scaled.shape[0]/timesteps), timesteps, x_valid_y0_scaled.shape[1])) # 테스트 데이터 셋
        x_valid_scaled_reshape = x_valid_scaled.reshape((int(x_valid_scaled.shape[0]/timesteps), timesteps, x_valid_scaled.shape[1])) # 테스트 데이터 셋
        x_test_scaled_reshape = x_test_scaled.reshape((int(x_test_scaled.shape[0]/timesteps), timesteps, x_test_scaled.shape[1])) # 테스트 데이터 셋

    def LSTM(self):
        self.checkFolder()
        with K.tf.device('/gpu:0'): # using gpu
            model = Sequential()
            model.add(LSTM(units=self.units, input_shape=self.inputshape,dropout=self.dropout))
            model.add(Dense(self.timesteps,activation='sigmoid'))
            model.compile(
                        loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'],
                        )
            model.summary()
            
            history = model.fit(train_scaled,train_label, epochs=self.epochs, batch_size=self.batchsize, verbose=self.verbose, shuffle=False)

            model.save('../models/LSTM.h5')
            return model

    def LSTM_CNN(self):
        self.checkFolder()
        with K.tf.device('/gpu:0'):
            model = Sequential()
            model.add(layers.Conv1D(filters=self.timesteps, kernel_size=n_features, activation='relu', padding='same', input_shape=(self.timesteps,78)))
            model.add(layers.MaxPooling1D(pool_size=self.timesteps, padding='same'))
            model.add(Dense(self.timesteps,activation='sigmoid'))
            model.add(LSTM(units=self.units, input_shape=self.inputshape,dropout=self.dropout)) # unit : 생성할 유닛수 = 출력갯수 / input_shape : 살펴볼 과거 데이터 수 / feature : 칼럼 개수
            model.add(Dense(self.units,activation='relu'))
            
            model.compile(
                        loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'],
                        )
            model.summary()
            history = model.fit(train_scaled,train_label, epochs=self.epochs, batch_size=self.batchsize, validation_data=(scaled, label), verbose=self.verbose, shuffle=False)
            model.save('../models/LSTM-CNN.h5')

    def LSTM_AE(self):
        self.checkFolder()
        with K.tf.device('/gpu:0'):
            model = models.Sequential()
            model.add(layers.LSTM(64, activation='relu', input_shape=self.inputshape, return_sequences=True))
            model.add(layers.LSTM(32, activation='relu', return_sequences=True))
            model.add(layers.LSTM(16, activation='relu', return_sequences=False))
            model.add(layers.RepeatVector(self.timesteps))
            # Decoder
            model.add(layers.LSTM(16, activation='relu', return_sequences=True))
            model.add(layers.LSTM(32, activation='relu', return_sequences=True))
            model.add(layers.LSTM(64, activation='relu', return_sequences=True))
            model.add(layers.TimeDistributed(layers.Dense(78)))
            
            model.compile(loss='mse', optimizer=optimizers.Adam(self.lr), metrics=['accuracy'],)
            history = model.fit(x_train_y0_scaled_reshape, x_train_y0_scaled_reshape,
                                    epochs=self.epochs, batch_size=self.batchsize,
                                    validation_data=(x_valid_y0_scaled_reshape, x_valid_y0_scaled_reshape))
            model.save('../models/LSTM-AE.h5')

    def GRU(self):
        self.checkFolder()
        with K.tf.device('/gpu:0'): # using gpu
            model = Sequential()
            model.add(GRU(units=self.units, input_shape=self.inputshape,dropout=self.dropout))
            model.add(Dense(self.timesteps,activation='sigmoid'))
            model.compile(
                        loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'],
                        )
            model.summary()
            history = model.fit(train_scaled,train_label, epochs=self.epochs, batch_size=self.batchsize, verbose=self.verbose, shuffle=False)
            model.save('../models/GRU.h5')
            return model

    def GRU_CNN(self):
        self.checkFolder()
        with K.tf.device('/gpu:0'):
            model = Sequential()
            model.add(layers.Conv1D(filters=self.timesteps, kernel_size=n_features, activation='relu', padding='same', input_shape=(self.timesteps,78)))
            model.add(layers.MaxPooling1D(pool_size=self.timesteps, padding='same'))
            model.add(Dense(self.timesteps,activation='sigmoid'))
            model.add(GRU(units=self.units, input_shape=self.inputshape,dropout=self.dropout)) # unit : 생성할 유닛수 = 출력갯수 / input_shape : 살펴볼 과거 데이터 수 / feature : 칼럼 개수
            model.add(Dense(self.units,activation='relu'))
            
            model.compile(
                        loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'],
                        )
            model.summary()
            history = model.fit(train_scaled,train_label, epochs=self.epochs, batch_size=self.batchsize, validation_data=(scaled, label), verbose=self.verbose, shuffle=False)
            model.save('../models/GRU-CNN.h5')

    def GRU_AE(self):
        self.checkFolder()
        with K.tf.device('/gpu:0'):
            model = models.Sequential()
            model.add(layers.GRU(64, activation='relu', input_shape=self.inputshape, return_sequences=True))
            model.add(layers.GRU(32, activation='relu', return_sequences=True))
            model.add(layers.GRU(16, activation='relu', return_sequences=False))
            model.add(layers.RepeatVector(self.timesteps))
            # Decoder
            model.add(layers.GRU(16, activation='relu', return_sequences=True))
            model.add(layers.GRU(32, activation='relu', return_sequences=True))
            model.add(layers.GRU(64, activation='relu', return_sequences=True))
            model.add(layers.TimeDistributed(layers.Dense(78)))
            
            model.compile(loss='mse', optimizer=optimizers.Adam(self.lr), metrics=['accuracy'],)
            history = model.fit(x_train_y0_scaled_reshape, x_train_y0_scaled_reshape,
                                    epochs=self.epochs, batch_size=self.batchsize,
                                    validation_data=(x_valid_y0_scaled_reshape, x_valid_y0_scaled_reshape))
            model.save('../models/GRU-AE.h5')            
            return model

    def detect(self, modelname, test_df):
        model = models.load_model(modelname)
        acc = model.predict(test_scaled)

        result = acc.reshape(acc.shape[0]*acc.shape[1],1)
        label = test_label.reshape(test_label.shape[0],1)[:result.shape[0]]
        
        result[result>=0.5] = 1
        result[result<0.5] = 0

        print("accuracy_score :",accuracy_score(label, result)*100)
        print("recall_score :",recall_score(label, result)*100)
        print("precision_score :",precision_score(label, result)*100)
        print("f1_score :",f1_score(label, result)*100)
