import string
from config import Config
from torch.utils.data import Dataset, DataLoader


def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def characters_length(data_path):
    text = load_data(data_path)
    return len(text)


def tokenize_text(text):
    text = text.lower()
    for punct in string.punctuation:
        text = text.replace(punct, f" {punct} ")
    return text.split()


def build_char_vocab(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    return vocab_size, chars


class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y
