from n_gram import BackoffNGramLanguageModel, plot_histograms
from config import Config
from RNN import RNNModel, RNN
import torch
import random


def main(n, model_choice, data_path):
    config = Config()

    if model_choice == 'ngram':

        print("Training N-gram Language Model...")
        sentences = BackoffNGramLanguageModel(data_path, n=n).load()
        random.shuffle(sentences)

        split = int(config.train_split * len(sentences))
        train_sents = sentences[:split]
        test_sents = sentences[split:]

        ngram_model = BackoffNGramLanguageModel(data_path, n=n, min_count=2)
        ngram_model.train(train_sents)
        stats = ngram_model.perplexity_with_stats(test_sents)

        print(f"\n=== n={n} ===")
        print("Perplexity:", stats["ppl"])
        print("Unigram fallback rate:", stats["unigram_fallback_rate"])
        print("Used-order counts:", dict(stats["used_order_counts"]))

        print("\nGenerated Text:")
        print(ngram_model.generate_text(
            max_length=500,
            method="random",
            seed_text="ROMEO:",
            temperature=0.8
        ))

        plot_histograms(stats, 3)

    if model_choice == 'RNN':
        print("\nTraining RNN Language Model...")

        rnn_trainer = RNN(data_path)

        text = rnn_trainer.load()
        vocab_size, chars = rnn_trainer.build_vocab(text)
        char2idx = rnn_trainer.vectorise_text(chars)
        idx2char = rnn_trainer.devectorise_text(char2idx)
        encoded = rnn_trainer.encode_text(text, char2idx)

        print("\nGenerated Text:")
        print(rnn_trainer.text_generate
              (idx2char, char2idx, vocab_size, start="ROMEO:", length=500, temperature=0.8))


data_path = "/Users/laibaqureshi/Desktop/Text Generation/Text-Generation/shakespeare_2.txt"
# Choose 'ngram' or 'RNN'
# main(n=3, model_choice='RNN', data_path=data_path)
main(n=3, model_choice='ngram', data_path=data_path)
