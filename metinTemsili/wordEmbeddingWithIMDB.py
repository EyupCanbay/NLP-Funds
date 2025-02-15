# import libraries
# set datasets
# text cleaning
# text tokenization
# wort to wktor definete model
# clustering KMeans K = 2
# PCA 50 -> 2 
# 2 boyutlu göreselleştirme

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import re

from sklearn.decomposition import PCA  
from sklearn.cluster import KMeans   

from gensim.models import Word2Vec # kullancağım modelim
from gensim.utils import simple_preprocess  # tokenlastırma için kullancam

df = pd.read_csv("./datasets/IMDB Dataset.csv")
#print(df.head(3))

documents = df["review"]
# metin temizleme
def clean_text(text):
    text = text.lower() # küçük harfe çevirme
    text = re.sub(r"\d+", "", text) #sayıları temizleme
    text = re.sub(r"[^\w\s]", "", text) # özel karakterleri temizleme
    text = " ".join([word for word in text.split() if len(re.sub(r"_", "", word)) > 2])
    text = simple_preprocess(text) #tokenization

    return text

tokenized_documents=[clean_text(doc) for doc in documents]

#modeli tanımlama
model = Word2Vec(sentences=tokenized_documents, vector_size=50, window=5, min_count=1, sg=0)
word_vector = model.wv

words = list(word_vector.index_to_key)[:1000]   # bütün hepsini alınca kapatıyor kendini
vector = [word_vector[word] for word in words]

#clustering KMeans K = 2
kmeans = KMeans(n_clusters=2)
kmeans.fit(vector)
clusters = kmeans.labels_ # 0 ve 1 lerden oluşan 2 tane küme oluşturdu
#print(clusters)

#PCA 
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vector) # 2 boyuta indermege yapıcak


# 2 boyutlu görselleştirme
plt.figure()
plt.scatter(reduced_vectors[:,0], reduced_vectors[:,1], c = clusters, cmap="viridis")



centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0], centers[:,1], c ="red", marker="x", s = 130, label = "Center")
plt.legend()


for i, word in enumerate(words):
    plt.text(reduced_vectors[i,0], reduced_vectors[i,1], word, fontsize=6)

plt.title("word'vec")


plt.show()
