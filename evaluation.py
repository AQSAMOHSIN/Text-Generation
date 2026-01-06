from n_gram import BackoffNGramLanguageModel, plot_histograms
from config import Config
from RNN import RNNModel, RNN
from LSTM import LSTMTextGen, LSTM
import torch
import random
from main import main
from config import Config
from dataset import CharDataset
from torch.utils.data import Dataset, DataLoader

data_path = "/Users/laibaqureshi/Desktop/Text Generation/Text-Generation/shakespeare_2.txt"
config = Config()
lstm_loader = LSTM(data_path)
rnn_loader = RNN(data_path)


def compare_generated_texts(data_path, prompt="ROMEO:", length=500, temp=0.8):
    print("Generated text through n-gram model:")
    n_gram_stats = main(n=3, model_choice='ngram', data_path=data_path,
                        prompt=prompt, length=length, temp=temp)

    print("Generated text through RNN model:")
    rnn_idx2char, rnn_char2idx, rnn_vocab_size, rnn_encoded = main(
        n=3, model_choice='RNN', data_path=data_path, prompt=prompt, length=length, temp=temp)

    print("Generated text through LSTM model:")
    lstm_idx2char, lstm_char2idx, lstm_vocab_size, lstm_encoded = main(
        n=3, model_choice='LSTM', data_path=data_path, prompt=prompt, length=length, temp=temp)

    return n_gram_stats, (rnn_idx2char, rnn_char2idx, rnn_vocab_size, rnn_encoded), (lstm_idx2char, lstm_char2idx, lstm_vocab_size, lstm_encoded)


n_gram_stats, (rnn_idx2char, rnn_char2idx, rnn_vocab_size, rnn_encoded), (lstm_idx2char,
                                                                          lstm_char2idx, lstm_vocab_size, lstm_encoded) = compare_generated_texts(data_path)


def perplexity_measure(encoded, vocab_size, model_type, stats=None, n=None):
    if model_type in ['RNN', 'LSTM']:
        split = int(config.train_split * len(encoded))
        train_data = encoded[:split]
        test_data = encoded[split:]

        test_dataset = CharDataset(test_data, config.sequence_length)

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,      # evaluation should be deterministic
            drop_last=False     # evaluate on all samples
        )

    if model_type == 'RNN':
        perplexity_stats = rnn_loader.perplexity_measure(
            vocab_size, test_loader)
        print("RNN:", perplexity_stats)

    if model_type == 'LSTM':
        perplexity_stats = lstm_loader.perplexity_measure(
            vocab_size, test_loader)
        print("LSTM:", perplexity_stats)

    if model_type == 'ngram':
        print(f"\n=== n={n} ===")
        print("Perplexity:", stats["ppl"])
        print("Unigram fallback rate:", stats["unigram_fallback_rate"])
        print("Used-order counts:", dict(stats["used_order_counts"]))
        plot_histograms(stats, 3)


print("\nEvaluating Perplexity on Test Set:")
perplexity_measure(rnn_encoded, rnn_vocab_size, model_type='RNN')
perplexity_measure(lstm_encoded, lstm_vocab_size, model_type='LSTM')
perplexity_measure(None, None, model_type='ngram', stats=n_gram_stats, n=3)

# evaluate diversity -> no gram uniqueness

# def ngram_repetition(text, n=3):
#     grams = [text[i:i+n] for i in range(len(text)-n)]
#     return len(grams) - len(set(grams))


# rnn_text = generate(rnn_model, "ROMEO:\n", 400, 0.8)
# lstm_text = generate(lstm_model, "ROMEO:\n", 400, 0.8)

# print("RNN repetitions:", ngram_repetition(rnn_text))
# print("LSTM repetitions:", ngram_repetition(lstm_text))

# # evaluate collapse -> repition loops


# def detects_loop(text):
#     for n in [3, 4, 5, 6]:
#         for i in range(len(text)-n*2):
#             if text[i:i+n] == text[i+n:i+2*n]:
#                 return True
#     return False


# print("RNN loop:", detects_loop(rnn_text))
# print("LSTM loop:", detects_loop(lstm_text))

# # evaluate creativity with temperature

# for temp in [0.6, 0.9, 1.2]:
#     print(f"\n====== TEMP {temp} - RNN ======")
#     print(generate(rnn_model, "ROMEO:\n", 250, temp))

#     print(f"\n====== TEMP {temp} - LSTM ======")
#     print(generate(lstm_model, "ROMEO:\n", 250, temp))
