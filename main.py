from n_gram import BackoffNGramLanguageModel, plot_histograms
from config import Config
from RNN import RNNModel, RNN
from LSTM import LSTMTextGen, LSTM
import torch
import random
from LLM import GPT2fineTuned


def main(n, model_choice, data_path, prompt, length, temp):
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

        # print(f"\n=== n={n} ===")
        # print("Perplexity:", stats["ppl"])
        # print("Unigram fallback rate:", stats["unigram_fallback_rate"])
        # print("Used-order counts:", dict(stats["used_order_counts"]))

        print("\nN-gram Generated Text:")
        gen_text = ngram_model.generate_text(
            max_length=length,
            method="greedy",
            seed_text=prompt,
            temperature=temp
        )
        print(gen_text)
        return stats, gen_text

        # plot_histograms(stats, 3)

    if model_choice == 'RNN':
        print("\nTraining RNN Language Model...")

        rnn_trainer = RNN(data_path)

        text = rnn_trainer.load()
        vocab_size, chars = rnn_trainer.build_vocab(text)
        char2idx = rnn_trainer.vectorise_text(chars)
        idx2char = rnn_trainer.devectorise_text(char2idx)
        encoded = rnn_trainer.encode_text(text, char2idx)

        print("\nRNN Generated Text:")
        gen_text = rnn_trainer.text_generate(
            idx2char, char2idx, vocab_size, start=prompt, length=length, temperature=temp)
        print(gen_text)

        return idx2char, char2idx, vocab_size, encoded, gen_text

    if model_choice == 'LSTM':
        print("\nTraining LSTM Language Model...")

        lstm_trainer = LSTM(data_path)

        text = lstm_trainer.load()
        vocab_size, chars = lstm_trainer.build_vocab(text)
        char2idx = lstm_trainer.vectorise_text(chars)
        idx2char = lstm_trainer.devectorise_text(char2idx)
        encoded = lstm_trainer.encode_text(text, char2idx)
        print("\nLSTM Generated Text:")
        gen_text = lstm_trainer.text_generate(
            idx2char, char2idx, vocab_size, start=prompt, length=length, temperature=temp)
        print(gen_text)

        return idx2char, char2idx, vocab_size, encoded, gen_text
    if model_choice == 'TRANSFORMER':
        gpt2 = GPT2fineTuned()
        print("\nLLM Generated Text:")
        gen_text = gpt2.generate_text(
            prompt,
            max_new_tokens=length,
            temperature=temp,
            top_p=0.9,
            top_k=50)
        print(gen_text)
        return gen_text


data_path = "/Users/laibaqureshi/Desktop/Text Generation/Text-Generation/shakespeare_2.txt"
# Choose 'ngram' or 'RNN' or 'LSTM' as model_choice
# main(n=3, model_choice='RNN', data_path=data_path)
# main(n=3, model_choice='ngram', data_path=data_path)
models = ['ngram', 'RNN', 'LSTM', 'TRANSFORMER']
prompt = "ROMEO:"
length = 500
temp = 0.8
for model in models:
    main(n=3, model_choice=model, data_path=data_path,
         prompt=prompt, length=length, temp=temp)
