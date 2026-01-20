from collections import defaultdict, Counter
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from config import Config
from dataset import *
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


class LSTMTextGen(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch):
        h = torch.zeros(self.num_layers, batch, self.hidden_dim).to(device)
        c = torch.zeros(self.num_layers, batch, self.hidden_dim).to(device)
        return (h, c)


class LSTM:
    def __init__(self, data_path):
        self.config = Config()
        self.data_path = data_path

    def load(self):
        text = load_data(self.data_path)
        print("Text characters length:", characters_length(self.data_path))
        return text

    def build_vocab(self, text):
        vocab_size, chars = build_char_vocab(text)
        print("Vocab size (unique chars):", vocab_size)
        print(chars)
        return vocab_size, chars

    def vectorise_text(self, chars):
        # Map characters to their indices in vocabulary.
        char2idx = {char: index for index, char in enumerate(chars)}

        print('{')
        for char, _ in zip(char2idx, range(20)):
            print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
        print('  ...\n}')
        return char2idx

    def devectorise_text(self, char2idx):
        # Map indices to their characters in vocabulary.
        idx2char = {index: char for char, index in char2idx.items()}
        print(idx2char)
        return idx2char

    def encode_text(self, text, char2idx):
        # Convert chars in text to indices.
        encoded = torch.tensor([char2idx[char]
                               for char in text], dtype=torch.long)
        print("Encoded shape:", encoded.shape)

        print("Original vs Encoded text:")
        print("Original text:", text[:200])
        print("Vectorised/Encoded text:", encoded[:200])
        return encoded

    def sample_logits(self, logits, temperature=1.0):
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()

    def text_generate(self, idx2char, char2idx, vocab_size, start="ROMEO:", length=500, temperature=0.7):
        model = LSTMTextGen(vocab_size).to(device)

        pretrained_path = os.getcwd() + "/Text-Generation/shakespeare_lstm.pth"

        # # load pretrained model
        # model.load_state_dict(torch.load(pretrained_path, map_location=device))

        # model.eval()

        ckpt = torch.load(pretrained_path, map_location=device)

        # pick the correct key that contains the state_dict
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "model_state" in ckpt:
                state_dict = ckpt["model_state"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                raise KeyError(
                    f"Checkpoint keys found: {list(ckpt.keys())}. Can't find model weights key.")
        else:
            # rare case: ckpt itself is already a state_dict
            state_dict = ckpt

        model.load_state_dict(state_dict)
        model.eval()

        input_ids = torch.tensor(
            [[char2idx[c] for c in start]], dtype=torch.long).to(device)
        hidden = model.init_hidden(1)

        generated = list(start)

        with torch.no_grad():
            for _ in range(length):
                logits, hidden = model(input_ids, hidden)
                last_logits = logits[0, -1]
                next_id = self.sample_logits(last_logits, temperature)
                generated.append(idx2char[next_id])

                input_ids = torch.tensor(
                    [[next_id]], dtype=torch.long).to(device)

        return "".join(generated)

    def load_state_dict_from_ckpt(self, path, device):
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict):
            for key in ["model_state_dict", "model_state", "state_dict"]:
                if key in ckpt:
                    return ckpt[key]
            # sometimes ckpt itself is a state_dict-like dict
            return ckpt
        return ckpt

    def perplexity_measure(self, vocab_size, loader):
        model = LSTMTextGen(vocab_size).to(device)

        pretrained_path = os.getcwd() + "/Text-Generation/shakespeare_lstm.pth"

        state_dict = self.load_state_dict_from_ckpt(pretrained_path, device)
        model.load_state_dict(state_dict)

        model.eval()
        model.to(device)

        loss_fn = nn.CrossEntropyLoss(reduction="sum")  # sum over tokens

        total_nll = 0.0
        total_tokens = 0

        for x, y in loader:
            x = x.to(device)  # [B, T]
            y = y.to(device)  # [B, T]
            B = x.size(0)

            hidden = model.init_hidden(B)
            if isinstance(hidden, tuple):
                hidden = tuple(h.to(device) for h in hidden)
            else:
                hidden = hidden.to(device)

            logits, _ = model(x, hidden)     # expected [B, T, V]
            # logits = logits.transpose(0, 1)

            total_nll += loss_fn(logits.reshape(-1, vocab_size),
                                 y.reshape(-1)).item()
            total_tokens += y.numel()

        avg_nll = total_nll / max(1, total_tokens)
        ppl = math.exp(avg_nll)
        bpc = avg_nll / math.log(2)

        return {"char_ppl": ppl, "bpc": bpc, "char_nll": avg_nll, "N_chars": total_tokens}

        # total_loss = 0.0
        # total_tokens = 0

        # with torch.no_grad():
        #     for inputs, targets in data_loader:
        #         inputs, targets = inputs.to(device), targets.to(device)
        #         batch_size, seq_len = inputs.size()
        #         hidden = model.init_hidden(batch_size)

        #         outputs, hidden = model(inputs, hidden)
        #         loss = criterion(outputs.view(-1, outputs.size(-1)),
        #                          targets.view(-1))
        #         total_loss += loss.item() * batch_size * seq_len
        #         total_tokens += batch_size * seq_len

        # avg_loss = total_loss / total_tokens
        # perplexity = torch.exp(torch.tensor(avg_loss))
        # return perplexity.item()
