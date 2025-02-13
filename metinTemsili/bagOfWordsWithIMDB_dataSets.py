# import libraries
# import datasers
# making test cleaning
# BoW
# definite vectorizer
# siwitch text to numeric
# show set of words
# show vector representation
# show words frequency

class Cleaning:
    def clean_text(self, text):        
        # Büyük/küçük harf dönüşümü
        text = text.lower()

        # Rakamları temizleme
        text = re.sub(r"\d+", "", text)

        # Özel karakterlerin kaldırılması
        text = re.sub(r"[^\w\s]", "", text)

        # Kısa kelimeleri temizleme (2 karakterden kısa olanları sil)
        text = " ".join([word for word in text.split() if len(word) > 2])

        # Stop words temizleme
        return self.clean_stop_words(text)

    def clean_stop_words(self, text):
        stopWordsEng = set(stopwords.words("english"))  # Stopwords listesini al
        filteredWords = [word for word in text.split() if word not in stopWordsEng]  # Stopwords çıkar
        return " ".join(filteredWords)  # Listeyi tekrar stringe çevir

    def write_txt(self, dosya_adi, veri_listesi):
        with open(dosya_adi, "w", encoding="utf-8") as dosya:
            for satir in veri_listesi:
                dosya.write(satir + "\n")


#kütüphanelerin içeri aktarılması
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re # metin etmizlemek için kullanılan kütüphane
from collections import Counter #frekans hesaplarken kullanıcağımız modul
import nltk 
from nltk.corpus import stopwords #stopwords temzlemek için kullanılan kütüphane
temizleme = Cleaning()


#verisetinin içeri aktarılması
df = pd.read_csv("./datasets/IMDB Dataset.csv")

#metin verilerini alalım
documents = df["review"]
labels = df["sentiment"]  # positive or negative
#print(documents, labels)

cleanedDoc = [temizleme.clean_text(row) for row in documents]

#bow

#vectorizer tanımlama
vectorizer = CountVectorizer()

#metin to numeric
x = vectorizer.fit_transform(cleanedDoc[:75])

#kelime kümesini göster
featuredNames = vectorizer.get_feature_names_out()
#print(featuredNames)

# vektör temsili
vectorTemsil = x.toarray()
#print(vectorTemsil)

#kelime frekanslarını gösterme
df_bow = pd.DataFrame(vectorTemsil, columns=featuredNames)

#kelime frekanslarını gösterme
wordCount=x.sum(axis = 0).A1
wordFreq = dict(zip(featuredNames, wordCount))


mostCommonWords = Counter(wordFreq).most_common(10)
print(mostCommonWords)