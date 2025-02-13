# countVektorizor içeriye aktarilir
# veri seti oluştur
# vektorizeri tanımla
# metni sayısal vektörlere cevir
# sonuçları incelenmesi vektör temsili

# countVektorizor içeriye aktarilir
from sklearn.feature_extraction.text import CountVectorizer

# veri seti oluştur
doc = {
    "kedi bahcede",
    "kedi evde"
}

# vektorizeri tanımla
vectorizer = CountVectorizer()

# metni sayısal vektörlere cevir
x = vectorizer.fit_transform(doc)

# sonuçları incelenmesi vektör temsili
featureNemes = vectorizer.get_feature_names_out()
print(featureNemes) # ['bahcede' 'evde' 'kedi']

# sonuçları incelenmesi vektör temsili
vectorTemsili = x.toarray()
print(vectorTemsili)# [[1 0 1]
 #                     [0 1 1]]
