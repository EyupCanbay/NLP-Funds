# import library
import pandas as pd
import nltk
from nltk.util import ngrams  # sklearn da kullanıbilirdim farklılık olsun diye bunu kullandım
from nltk.tokenize import word_tokenize #tokenizasyon için
from collections import Counter

# örnek veri seti oluştur
corpus = [
    "I ate apple",
    "I ate sandwich",
    "I ate dinner",
    "I took apple",
    "I took apple",
    "We took apple",
    "The cat is sleeping",
    "She is sleeping now",
    "They are sleeping late",
    "He love popcorn",
    "I love popcorn",
    "She love popcorn",
    "We went outside",
    "He went outside",
    "She went outside",
]



"""
Problem tanımı
    dil modeli yapmak istiyorum
    amac 1 kelimeden sonra gelicek kelimeyi tahmin etmek: metin türetmek oluşturmak
    bunun için n gram dil modeli kullanıcaz
"""


# verileri token haline getir
tokens = [word_tokenize(sentence.lower()) for sentence in corpus]
#print(tokens)

# ikili kelime grupları oluştur (bigram)
bigrams = []
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list, 2)))
bigrams_freq = Counter(bigrams)


# print(bigrams_freq ," \n")
#trigram
trigrams = []
for token_list in tokens:
    trigrams.extend(list(ngrams(token_list, 3)))

trigram_freq = Counter(trigrams)
# print(trigram_freq)


#model testing

#ı ate bi gramından sonrra you gelecek kelimeye bakalım

bigram = ("i", "ate") # hedef bigram

prob_apple = trigram_freq[("i","ate", "apple")] / bigrams_freq[bigram]

print("apple kelimelisinin olma olasılığı" , prob_apple)

# ı love popcorn olma olasılığı

bigram = ("i", "love") # hedef bigram

prob_apple = trigram_freq[("i","love", "popcorn")] / bigrams_freq[bigram]

print("popcorn kelimelisinin olma olasılığı" , prob_apple)








