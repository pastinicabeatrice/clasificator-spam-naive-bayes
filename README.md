# Clasificator de spam - Naive Bayes

Clasificator de mesaje spam / non-spam, implementat de la zero cu algoritmul Naive Bayes multinomial.

## Cum funcționează

Pentru un mesaj, modelul calculează pentru fiecare clasă:

```
log P(clasa) + Σ log P(cuvânt | clasa)
```

și alege clasa cu scorul cel mai mare.

- probabilitățile cuvintelor se învață din frecvențele din mesajele de antrenare
- netezirea Laplace (+1) evită probabilitatea zero pentru cuvintele care nu apar într-o clasă
- se lucrează cu logaritmi, ca produsul multor probabilități mici să nu devină 0

## Date

`mesaje.csv` conține 40 de mesaje scurte în limba română, etichetate `spam` sau `ham`.

## Rulare

Necesită Python 3.8 sau mai nou, fără alte dependențe.

```
python main.py
python main.py "Ai castigat o excursie gratuita" "Vii diseara la film?"
python -m unittest
```

`main.py` afișează acuratețea pe un set de test separat, cuvintele cele mai tipice pentru spam
și clasificarea mesajelor date ca argumente.
