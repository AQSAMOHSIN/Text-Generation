import math
import random
from collections import defaultdict, Counter
import re


class NGramLanguageModel:
    def __init__(self, n=3):
        """
        Initialize n-gram language model
        n: order of the n-gram (e.g., 3 for trigram)
        """
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocabulary = set()
        self.total_words = 0
        self.smoothing = 0.01  # Add-k smoothing parameter

    def tokenize(self, text):
        """
        Simple tokenization - split on whitespace and punctuation
        """
        # Convert to lowercase and split on whitespace
        text = text.lower()
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Keep punctuation as separate tokens &  tokenize the text into list of words
        tokens = re.findall(r'\w+|[.!?,;]', text)
        return tokens

    def get_ngrams(self, tokens):
        """
        Generate n-grams from token list
        """
        # Add start and end tokens
        padded = ['<s>'] * (self.n - 1) + tokens + ['</s>']

        ngrams = []
        for i in range(len(padded) - self.n + 1):
            context = tuple(padded[i:i + self.n - 1])
            word = padded[i + self.n - 1]
            ngrams.append((context, word))

        return ngrams

    def load(self, corpus):
        """
        Train the model on a corpus
        """
        print(f"Training {self.n}-gram model on {corpus}...")

        with open(corpus, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split into sentences (simple approach)
        sentences = text.lower().splitlines()
        return sentences

    def train(self, sentences):
        for sentence in sentences:
            if sentence.strip():
                tokens = self.tokenize(sentence)
                self.vocabulary.update(tokens)

                # Get n-grams
                ngrams = self.get_ngrams(tokens)

                # Count n-grams and contexts
                for context, word in ngrams:
                    self.ngram_counts[(context, word)] += 1
                    self.context_counts[context] += 1
                    self.total_words += 1

        self.vocabulary.add('<s>')
        self.vocabulary.add('</s>')
        self.vocabulary.add('<unk>')

        print(f"Training complete. Vocabulary size: {len(self.vocabulary)}")
        print(f"Total n-grams: {len(self.ngram_counts)}")

    def word_probability(self, word, context):
        """
        Calculate probability of word given context
        Using add-k smoothing
        """
        if context not in self.context_counts:
            # Unseen context - use uniform distribution
            return 1.0 / len(self.vocabulary)

        ngram_count = self.ngram_counts.get((context, word), 0)
        context_count = self.context_counts[context]

        # Add-k smoothing
        probability = (ngram_count + self.smoothing) / \
            (context_count + self.smoothing * len(self.vocabulary))

        return probability

    def generate_word(self, context, method='random'):
        """
        Generate next word given context
        method: 'random' for probabilistic, 'greedy' for most likely
        """
        if method == 'greedy':
            # Return most likely word
            best_word = None
            best_prob = 0

            for word in self.vocabulary:
                prob = self.word_probability(word, context)
                if prob > best_prob:
                    best_prob = prob
                    best_word = word

            return best_word

        else:  # Random sampling
            # Build probability distribution
            words = list(self.vocabulary)
            probs = [self.word_probability(word, context) for word in words]

            # Normalize probabilities
            total = sum(probs)
            probs = [p / total for p in probs]

            # Sample from distribution
            return random.choices(words, weights=probs)[0]

    def generate_text(self, max_length=50, method='random', seed_text=None):
        """
        Generate text using the trained model
        """
        generated = []

        # Initialize context
        if seed_text:
            tokens = self.tokenize(seed_text)
            generated.extend(tokens)
            # Use last n-1 tokens as context
            if len(tokens) >= self.n - 1:
                context = tuple(tokens[-(self.n - 1):])
            else:
                # Pad with start tokens if needed
                context = tuple(['<s>'] * (self.n - 1 - len(tokens)) + tokens)
        else:
            context = tuple(['<s>'] * (self.n - 1))

        # Generate words
        for _ in range(max_length):
            word = self.generate_word(context, method)

            if word == '</s>':
                break

            if word not in ['<s>', '<unk>']:
                generated.append(word)

            # Update context
            context = context[1:] + (word,)

        return ' '.join(generated)


ngram = NGramLanguageModel(n=3)
sentences = ngram.load(
    '/Users/laibaqureshi/Desktop/Text Generation/shakespeare.txt')
ngram.train(sentences)


seed = "to be or not to be"
text = ngram.generate_text(
    max_length=100, method='greedy', seed_text=None)
print(text)


"""will take a chunk from the training dataset as testing purpose and see how well the model performs
testing based on how similar does the generated text look like to the testing data
perplexity calculation can also be done to evaluate the model performance??"""


# Debugging and inspection
# print(f"Number of sentences: {len(sentences)}")
# print("First 3 sentences:")
# for i in range(3):
#     print(f"{i+1}: {sentences[i].strip()}")
# ngram.train(sentences)
# i = 0
# for key in ngram.context_counts:
#     print(key, ngram.context_counts[key])
#     i = i+1
#     if i == 5:
#         break

# i = 0
# for key in ngram.ngram_counts:
#     print(key, ngram.ngram_counts[key])
#     i = i+1
#     if i == 5:
#         break
# print(ngram.ngram_counts[:10])
# print(ngram.context_counts[:10])


# tokens = ngram.tokenize("""Natural Language Processing (NLP) is a subfield of artificial intelligence that bridges the gap between computers and human language. It enables machines to understand, interpret, and generate human language, which is crucial for applications like chatbots, automated translation, sentiment analysis, and text generation.""")
# print(tokens)
# ngrams = ngram.get_ngrams(tokens)
# print(ngrams)
