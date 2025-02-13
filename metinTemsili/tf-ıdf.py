# import libraries
# örnek belge oluştur
#vectorizer tanımla
# metinleri sayısal hale cevir
#kelime kümesini incele
# vektör temsilinş şncele
#ortalama tf ıdf değerlenine bakalım


import pandas as pd
import numpy as nm
from sklearn.feature_extraction.text import TfidfVectorizer

# örnek belge oluştur
documents = [
    "Köpek çok tatlı bir hayvandır",
    "Köpek ve kuşlar çok tatlı hayvanlardır",
    "Inekler süt üretirler"
]

#vectorizer tanımla
tfidf_vectorizer = TfidfVectorizer()

# metinleri sayısal hale cevir
x = tfidf_vectorizer.fit_transform(documents)

#kelime kümesini incele
featuresName = tfidf_vectorizer.get_feature_names_out()
print(featuresName)

# vektör temsilini incele
vektor_temsin = x.toarray()
print(vektor_temsin)
df_tfidf = pd.DataFrame(vektor_temsin, columns=featuresName)
print(df_tfidf)

#ortalama tf ıdf değerlenine bakalım
tf_idf = df_tfidf.mean(axis=0)
#print(tf_idf)

# Sıralı TF-IDF
sorted_tf_idf = tf_idf.sort_values(ascending=False)

# Sonucu göster
print(sorted_tf_idf)