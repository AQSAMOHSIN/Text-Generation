import os

from matplotlib.pyplot import text


class Config():
    # data_path = os.getcwd() + "/Text-Generation/shakespeare_2.txt"
    data_path = os.getcwd() + "/shakespeare_2.txt"

    RANDOM_SEED = 42
    train_split = 0.85
    test_split = 0.15

    # For n-gram; k-smoothing parameter
    smoothing = 0.1

    # For RNN/LSTM;
    learning_rate = 0.001
    epochs = 100
    patience = 4
    sequence_length = 100
    batch_size = 64
