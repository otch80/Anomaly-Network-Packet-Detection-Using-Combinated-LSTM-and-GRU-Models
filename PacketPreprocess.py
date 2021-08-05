import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Preprocess:
    def __init__(self):
        print(">>> Packet Preprocess 객체 생성 완료")

    def loadCsv(self, target_df="dataset/train_df.csv", label="dataset/train_df.csv"):
        self.target_df = pd.read_csv(target_df, index_col=2)
        self.label = pd.read_csv(label)
        print(">>> Train, Test, Label csv 로드 완료")

    def missingValue(self):
        self.target_df = self.target_df.replace([np.inf, -np.inf], np.nan)
        self.target_df['Flow Byts/s'].fillna(0.0,inplace=True)
        self.target_df['Flow Pkts/s'].fillna(0.0,inplace=True)

        self.target_df.loc[self.target_df['Label']!='Benign','Label'] = 1
        self.target_df.loc[self.target_df['Label']=='Benign','Label'] = 0
        print(">>> 결측치 제거 완료")

    def normalizeToModel(self, timesteps=10):
        scaler = MinMaxScaler(feature_range=(0,1))
        values = self.target_df.values

        values = values.astype('float64')
        scaled = scaler.fit_transform(values)
        train = scaled[:-6].reshape((int(scaled.shape[0]/timesteps), timesteps, scaled.shape[1]))

        train_df = train[:,:,:-1]
        label_df = train[:,:,-1]
        print(">>> 정규화 완료")

        return train_df, label_df

    def normalizeToAE(self, timesteps=10):
        # 정상패킷 추출
        x_train = self.target_df.loc[self.target_df['Label']==0] 
        x_train = x_train.astype('float64').values

        scaler = MinMaxScaler(feature_range=(0,1))
        x_train = scaler.fit_transform(x_train)

        return x_train

    def reshapeToMultyShape(self, target_df, label_df):
        target_df = target_df[:-6].reshape(int(target_df.shape[0]/self.timesteps), self.timesteps, target_df.shape[1])
        label_df = label_df[:-6].reshape(int(label_df.shape[0]/self.timesteps), self.timesteps)
        print(">>> 차원 변환 완료")

        return target_df, label_df

    def __del__(self):
        print(">>> Packet Preprocess 객체 제거 완료")