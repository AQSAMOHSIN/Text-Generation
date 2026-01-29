import math
import random
import string
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from dataset import *
from config import Config


class BackoffNGramLanguageModel:
    def __init__(self, data_path, n=3, min_count=1):
        self.config = Config()
        self.n = n
        self.data_path = data_path

        # smaller than 1.0 is usually better (defined in config)
        self.smoothing = self.config.smoothing
        self.min_count = min_count          # replace rare words with <unk> if >1

        # counts_by_order = {k0 : (), k1 : (), ..}
        # eg: for k=3 - {3 : {((w1, w2), w3): count, ((wx, wy), wz): count, ...}
        # like this for k = 1, 2, ..., n
        self.counts_by_order = {k: defaultdict(int) for k in range(1, n + 1)}

        # and context_counts_by_order[k][context]
        # context_counts_by_order = {k0 : (), k1 : (), ..}
        # eg: for k=3 - {3 : {(w1, w2): count, (wx, wy): count, ...}

        self.context_counts_by_order = {
            k: defaultdict(int) for k in range(1, n + 1)}

        self.vocabulary = set()
        self.word_counts = Counter()

    def tokenize(self, text):
        return (tokenize_text(text))
        # text = text.lower()
        # for punct in string.punctuation:
        #     text = text.replace(punct, f" {punct} ")
        # return text.split()

    def load(self):
        text = load_data(self.data_path)
        return text.lower().splitlines()
        # with open(corpus, "r", encoding="utf-8") as f:
        #     text = f.read()
        # return text.lower().splitlines()

    def _build_vocab(self, sentences):
        wc = Counter()
        for s in sentences:
            if not s.strip():
                continue
            wc.update(self.tokenize(s))

        self.word_counts = wc

        # keep words above threshold, map rest to <unk>
        vocab = {w for w, c in wc.items() if c >= self.min_count}
        vocab.update({"<s>", "</s>", "<unk>"})
        self.vocabulary = vocab

    def _map_unk(self, tokens):
        return [t if t in self.vocabulary else "<unk>" for t in tokens]

    def train(self, sentences):
        # 2-pass: build vocab first (so <unk> mapping is consistent)
        self._build_vocab(sentences)

        for s in sentences:
            if not s.strip():
                continue
            tokens = self._map_unk(self.tokenize(s))

            # build all orders 1..n
            for k in range(1, self.n + 1):
                padded = ["<s>"] * (k - 1) + tokens + ["</s>"]
                for i in range(len(padded) - k + 1):
                    context = tuple(padded[i:i + k - 1])  # length k-1
                    word = padded[i + k - 1]
                    self.counts_by_order[k][(context, word)] += 1
                    self.context_counts_by_order[k][context] += 1

        print(f"Training complete. n={self.n}, vocab={len(self.vocabulary)}")

    def _smoothed_prob(self, k, word, context):
        # add-k smoothing at a given order
        V = len(self.vocabulary)
        c_ng = self.counts_by_order[k].get((context, word), 0)
        c_ctx = self.context_counts_by_order[k].get(context, 0)
        return (c_ng + self.smoothing) / (c_ctx + self.smoothing * V)

    def generate_word_backoff(self, full_context, method="random", temperature=1.0):
        """
        full_context: tuple length (n-1)
        method: "random" or "greedy"
        temperature: >1 flatter, <1 sharper (only used for random)
        """
        # Don't end immediately
        disallowed = {"<s>"}  # allow </s> so sentences can end naturally

        words = [w for w in self.vocabulary if w not in disallowed]

        # compute probs with backoff
        probs = []
        used_orders = []
        for w in words:
            p, used_k = self.word_probability_backoff(w, full_context)
            probs.append(p)
            used_orders.append(used_k)

        if method == "greedy":
            idx = max(range(len(words)), key=lambda i: probs[i])
            return words[idx]

        # random sampling with temperature
        if temperature <= 0:
            temperature = 1.0

        # apply temperature in log-space (stable-ish)
        logps = [math.log(p) / temperature for p in probs]
        m = max(logps)
        expps = [math.exp(lp - m) for lp in logps]
        Z = sum(expps)
        norm_probs = [e / Z for e in expps]

        return random.choices(words, weights=norm_probs, k=1)[0]

    def generate_text(self, max_length=50, method="random", seed_text=None, temperature=1.0):
        """
        Generate text from the trained model.
        """
        if seed_text:
            seed_tokens = self._map_unk(self.tokenize(seed_text))
        else:
            seed_tokens = []

        generated = list(seed_tokens)

        # initial context (n-1 tokens)
        if len(seed_tokens) >= self.n - 1:
            context = tuple(seed_tokens[-(self.n - 1):])
        else:
            context = tuple(
                ["<s>"] * (self.n - 1 - len(seed_tokens)) + seed_tokens)

        for _ in range(max_length):
            w = self.generate_word_backoff(
                context, method=method, temperature=temperature)

            if w == "</s>":
                break

            if w not in {"<s>"}:
                generated.append(w)

            # slide the window
            if self.n > 1:
                context = context[1:] + (w,)
            else:
                context = tuple()  # unigram has empty context

        return " ".join(generated)

    def word_probability_backoff(self, word, full_context):
        """
        full_context: tuple length (self.n-1)
        Backoff rule: use highest order with count>0, else go lower, else unigram.
        Returns: (prob, used_order)
        """
        if word not in self.vocabulary:
            word = "<unk>"

        # try n, n-1, ..., 2 using "seen ngram?" (count>0)
        for k in range(self.n, 1, -1):
            ctx = tuple(full_context[-(k - 1):]) if (k - 1) > 0 else tuple()
            if self.counts_by_order[k].get((ctx, word), 0) > 0:
                return self._smoothed_prob(k, word, ctx), k

        # unigram fallback (order 1)
        return self._smoothed_prob(1, word, tuple()), 1

    def perplexity_with_stats(self, sentences):
        log_prob = 0.0
        N = 0

        # stats
        oov_per_sentence = []
        used_order_counts = Counter()
        unigram_probs = []

        for s in sentences:
            if not s.strip():
                continue

            raw_tokens = self.tokenize(s)
            oov_count = sum(1 for t in raw_tokens if t not in self.vocabulary)
            oov_per_sentence.append(oov_count)

            tokens = self._map_unk(raw_tokens)

            # build n-grams for evaluation (only order n, but we backoff internally)
            padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
            for i in range(len(padded) - self.n + 1):
                context = tuple(padded[i:i + self.n - 1])
                word = padded[i + self.n - 1]

                prob, used_k = self.word_probability_backoff(word, context)
                used_order_counts[used_k] += 1
                if used_k == 1:
                    unigram_probs.append(prob)

                log_prob += math.log(prob)
                N += 1

        ppl = math.exp(-log_prob / max(N, 1))
        full_text = "\n".join([s for s in sentences if s.strip()])
        total_chars = len(full_text)
        bpc = (-log_prob / math.log(2)) / max(total_chars, 1)

        stats = {
            "ppl": ppl,
            "bpc": bpc,
            "N_tokens": N,
            "used_order_counts": used_order_counts,
            "unigram_fallback_rate": (used_order_counts[1] / max(N, 1)),
            "avg_unigram_prob_when_fallback": (sum(unigram_probs) / max(len(unigram_probs), 1)),
            "oov_sentence_list": oov_per_sentence,
            "unigram_probs_list": unigram_probs,
        }
        return stats


def plot_histograms(stats, n):
    # 1) Histogram: OOV tokens per sentence
    oov = stats["oov_sentence_list"]
    if oov:
        plt.figure()
        bins = range(0, max(oov) + 2)
        plt.hist(oov, bins=bins, edgecolor="black")
        plt.xlabel("Number of OOV tokens in a sentence")
        plt.ylabel("Number of sentences")
        plt.title("OOV frequency per sentence")
        plt.show()

    # 2) Bar chart: how often we used each order (n..1)
    used = stats["used_order_counts"]
    plt.figure()
    orders = list(range(1, n + 1))
    counts = [used.get(k, 0) for k in orders]
    plt.bar([str(k) for k in orders], counts)
    plt.xlabel("Used n-gram order")
    plt.ylabel("Token count")
    plt.title("Backoff usage: which order was used")
    plt.show()

    # 3) Histogram of unigram probabilities used during unigram fallback
    uni_probs = stats["unigram_probs_list"]
    if uni_probs:
        plt.figure()
        plt.hist(uni_probs, bins=50, edgecolor="black")
        plt.xlabel("Unigram probability used (when backoff hit order=1)")
        plt.ylabel("Count")
        plt.title("Distribution of unigram fallback probabilities")
        plt.show()


path = "/Users/laibaqureshi/Desktop/Text Generation/Text-Generation/shakespeare_2.txt"

sentences = BackoffNGramLanguageModel(path, n=3).load()
random.shuffle(sentences)  # important: avoid “last 20% is different”

split = int(0.8 * len(sentences))
train_sents = sentences[:split]
test_sents = sentences[split:]

for n in [1, 2, 3]:
    model = BackoffNGramLanguageModel(path, n=n, min_count=2)
    model.train(train_sents)
    stats = model.perplexity_with_stats(test_sents)

    print(f"\n=== n={n} ===")
    print("Perplexity:", stats["ppl"])
    print("Unigram fallback rate:", stats["unigram_fallback_rate"])
    print("Used-order counts:", dict(stats["used_order_counts"]))

    print(model.generate_text(
        max_length=50,
        method="random",
        seed_text="from fairest creatures",
        temperature=1.0
    ))

    print(model.generate_text(
        max_length=50,
        method="greedy",
        seed_text="from fairest creatures"
    ))

    plot_histograms(stats, n)
