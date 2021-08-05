from PacketPreprocess import Preprocess
import packetpreprocess
import detectmodels

if __name__ == "__main__":
    preprocess = packetpreprocess.Preprocess()
    detectmodel = detectmodels.DetectModels()

    preprocess.loadCsv("dataset/train_df.csv", "dataset/label_df.csv")
    preprocess.missingValue()
    train_df, label_df = preprocess.normalizeToModel(timesteps=10)
    train_df, label_df = preprocess.reshapeToMultyShape(train_df, label_df)

    train_ae_df, test_ae_df = preprocess.normalizeToAE(timesteps=10)
    train_ae_df, test_ae_df = preprocess.reshapeToMultyShape(train_ae_df, test_ae_df)

    train_df, test_df = detectmodel.trainTestSplit(train_df)
    train_ae_df, test_ae_df = detectmodel.trainTestSplit(train_ae_df)

    lstm, lstm_history = detectmodel.LSTM()
    lstm_cnn, lstm_cnn_history = detectmodel.LSTM_CNN()
    lstm_ae, lstm_ae_history = detectmodel.LSTM_AE()
    gru, gru_history = detectmodel.GRU()
    gru_cnn, gru_cnn_history = detectmodel.GRU_CNN()
    gru_ae, gru_ae_history = detectmodel.GRU_AE()