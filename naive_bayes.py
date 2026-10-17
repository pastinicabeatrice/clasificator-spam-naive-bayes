import math
import re
from collections import Counter, defaultdict


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class NaiveBayes:
    def fit(self, texts, labels):
        self.class_counts = Counter(labels)
        self.word_counts = defaultdict(Counter)
        for text, label in zip(texts, labels):
            self.word_counts[label].update(tokenize(text))
        self.vocabulary = {word for counts in self.word_counts.values() for word in counts}
        self.totals = {label: sum(counts.values()) for label, counts in self.word_counts.items()}
        return self

    def log_probability(self, words, label):
        prior = math.log(self.class_counts[label] / sum(self.class_counts.values()))
        denominator = self.totals[label] + len(self.vocabulary)
        return prior + sum(math.log((self.word_counts[label][word] + 1) / denominator) for word in words)

    def scores(self, text):
        words = [word for word in tokenize(text) if word in self.vocabulary]
        return {label: self.log_probability(words, label) for label in self.class_counts}

    def predict(self, text):
        scores = self.scores(text)
        return max(scores, key=scores.get)

    def top_words(self, label, count=10):
        other = [l for l in self.class_counts if l != label]
        ratio = lambda word: (self.word_counts[label][word] + 1) / (sum(self.word_counts[l][word] for l in other) + 1)
        return sorted(self.vocabulary, key=ratio, reverse=True)[:count]
