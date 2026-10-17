import csv
import random
import sys

from naive_bayes import NaiveBayes


def load(path="mesaje.csv"):
    with open(path, encoding="utf-8") as file:
        return [(row["mesaj"], row["eticheta"]) for row in csv.DictReader(file)]


def evaluate(model, data):
    correct = sum(model.predict(text) == label for text, label in data)
    return correct / len(data)


def main():
    data = load()
    random.Random(7).shuffle(data)
    split = int(len(data) * 0.75)
    train, test = data[:split], data[split:]

    model = NaiveBayes().fit(*zip(*train))
    print(f"Antrenare pe {len(train)} mesaje, test pe {len(test)}")
    print(f"Acuratete pe setul de test: {evaluate(model, test):.0%}")
    print("Cuvinte tipice pentru spam:", ", ".join(model.top_words("spam")))

    model = NaiveBayes().fit(*zip(*data))
    messages = sys.argv[1:] or [
        "Ai castigat un premiu, apasa pe link",
        "Ne vedem la curs maine?",
    ]
    print()
    for message in messages:
        print(f"[{model.predict(message)}] {message}")


if __name__ == "__main__":
    main()
