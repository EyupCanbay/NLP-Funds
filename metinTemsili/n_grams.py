# import lirary
# örnek metin
# n gram implemantanyonu  unigram bigram trigram 
# sonuçların incelenmesi


#kütüphanelerin kurulması
from sklearn.feature_extraction.text import CountVectorizer

# örnek metin
documents = [
    "Bu çalışma NGram çalışmasıdır",
    "Bu çalışma doğal dil işleme çalışmasıdır"
]

# n gram implemantanyonu  unigram bigram trigram 
vectorizer_unigram = CountVectorizer(ngram_range=(1,1))
vectorizer_bigram = CountVectorizer(ngram_range=(2,2))
vectorizer_trigram = CountVectorizer(ngram_range=(3,3))
 
#unigram
x_unigram = vectorizer_unigram.fit_transform(documents)
unigramFeatures = vectorizer_unigram.get_feature_names_out()
print(unigramFeatures)

# bigram
x_bigram = vectorizer_bigram.fit_transform(documents)
bigramFeatures = vectorizer_bigram.get_feature_names_out()
print(bigramFeatures)

#trigram
x_trigram = vectorizer_trigram.fit_transform(documents)
trigramFeatures = vectorizer_trigram.get_feature_names_out()
print(trigramFeatures)



