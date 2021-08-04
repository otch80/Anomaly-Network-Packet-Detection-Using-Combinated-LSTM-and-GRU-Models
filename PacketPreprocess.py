import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Preprocess:
    def __init__(self):
        print(">>> Packet Preprocess 객체 생성 완료")

    def loadCsv(self, train="dataset/train_df.csv", test="dataset/train_df.csv", label="dataset/train_df.csv"):
        self.train = pd.read_csv(train, index_col=2)
        self.test = pd.read_csv(test)
        self.label = pd.read_csv(label)
        print(">>> Train, Test, Label csv 로드 완료")

    def missingValue(self):
        self.train_df = self.train_df.replace([np.inf, -np.inf], np.nan)
        self.train_df['Flow Byts/s'].fillna(0.0,inplace=True)
        self.train_df['Flow Pkts/s'].fillna(0.0,inplace=True)

        self.train_df.loc[self.train_df['Label']!='Benign','Label'] = 1
        self.train_df.loc[self.train_df['Label']=='Benign','Label'] = 0

    def normalizeToLstm(self, timesteps=10):
        scaler = MinMaxScaler(feature_range=(0,1))
        values = self.train_df.values

        values = values.astype('float64')
        scaled = scaler.fit_transform(values)
        train = scaled[:-6].reshape((int(scaled.shape[0]/timesteps), timesteps, scaled.shape[1]))

        train_df = train[:,:,:-1]
        label_df = train[:,:,-1]
        return zip(train_df, label_df)

    def __del__(self):
        print(">>> Packet Preprocess 객체 제거 완료")