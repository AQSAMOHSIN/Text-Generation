import numpy as np
from sacrebleu.metrics import BLEU
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

data_path = """/Users/laibaqureshi/Desktop/Text Generation/Text-Generation/shakespeare_2.txt"""
config = Config()
lstm_loader = LSTM(data_path)
rnn_loader = RNN(data_path)


def compare_generated_texts(data_path, prompt="ROMEO:", length=500, temp=0.8):
    print("Generated text through n-gram model:")
    n_gram_stats, gen_text = main(n=3, model_choice='ngram', data_path=data_path,
                                  prompt=prompt, length=length, temp=temp)

    print("Generated text through RNN model:")
    rnn_idx2char, rnn_char2idx, rnn_vocab_size, rnn_encoded, gen_text = main(
        n=3, model_choice='RNN', data_path=data_path, prompt=prompt, length=length, temp=temp)

    print("Generated text through LSTM model:")
    lstm_idx2char, lstm_char2idx, lstm_vocab_size, lstm_encoded, gen_text = main(
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


def self_bleu_word(texts, max_refs=None, seed=0, lowercase=True):
    """
    Word-level Self-BLEU (BLEU-4 by default in sacrebleu).
    Higher Self-BLEU = samples are more similar (less diverse).
    Lower Self-BLEU = more diverse generations.
    """
    rng = np.random.default_rng(seed)
    bleu = BLEU(lowercase=lowercase, effective_order=True)

    scores = []
    N = len(texts)

    for i in range(N):
        hyp = texts[i]
        refs = [texts[j] for j in range(N) if j != i]

        # With N=10, you can just use all refs; max_refs is optional
        if max_refs is not None and len(refs) > max_refs:
            idx = rng.choice(len(refs), size=max_refs, replace=False)
            refs = [refs[k] for k in idx]

        s = bleu.sentence_score(hyp, refs).score  # 0..100
        scores.append(s)

    return float(np.mean(scores)), float(np.std(scores)), scores


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_k_samples(model_choice, data_path, prompt, length, temp, k=10, base_seed=123):
    samples = []
    for i in range(k):
        set_seed(base_seed + i)

        if model_choice == 'ngram':
            _, out = main(n=3, model_choice=model_choice, data_path=data_path,
                          prompt=prompt, length=length, temp=temp)
        else:
            _, _, _, _, out = main(n=3, model_choice=model_choice, data_path=data_path,
                                   prompt=prompt, length=length, temp=temp)
        generated_text = out

        samples.append(generated_text)
    return samples


k = 10
prompt = "ROMEO:"
length = 500
temp = 0.8

ngram_samples = generate_k_samples(
    "ngram", data_path, prompt, length, temp, k=k)
rnn_samples = generate_k_samples("RNN",   data_path, prompt, length, temp, k=k)
lstm_samples = generate_k_samples(
    "LSTM",  data_path, prompt, length, temp, k=k)

for name, samples in [("n-gram", ngram_samples), ("RNN", rnn_samples), ("LSTM", lstm_samples)]:
    mean_sb, std_sb, _ = self_bleu_word(samples, lowercase=True)
    print(f"{name} Self-BLEU (word, BLEU-4) : {mean_sb:.2f} ± {std_sb:.2f}")


# evaluate diversity -> no gram uniqueness

# def ngram_repetition(text, n=3):
#     grams = [text[i:i+n] for i in range(len(text)-n)]
#     return len(grams) - len(set(grams))


# rnn_text = generate(rnn_model, "ROMEO:\n", 400, 0.8)
# lstm_text = generate(lstm_model, "ROMEO:\n", 400, 0.8)

# print("RNN repetitions:", ngram_repetition(rnn_text))
# print("LSTM repetitions:", ngram_repetition(lstm_text))
