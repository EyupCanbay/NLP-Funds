# ingilizce stop words analizi  (nltk)
import nltk
from nltk.corpus import stopwords

#nltk.download("stopwords") # farklı dillerde en cok kullanılan stopwords içeren veri seti


stopWordsEng = set(stopwords.words("english"))
print(stopWordsEng)
# example text
text  = "There are some examples for handling stop words from some text."
textList = text.split()
filteredWords = [word for word in textList if word.lower() not in stopWordsEng]
print(" ".join(filteredWords))

# türkce stop words analizi (nltk)
stopWordsTR = set(stopwords.words("turkish"))
#örnek metin
metin = "merhaba arkadaşlar çok güzel bir ders işliyoruz biz. Bu  ders"
metinList = metin.split()
filteredWords = [word for word in metinList if word.lower() not in stopWordsTR]
print(" ".join(filteredWords))






# kütüphanesiz stop words cıkarımı
tr_stopwords=['fakat','lakin','ancak','acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani']
metin  = "fakat lakin sen bana nasıl böyle dersin haltsiz"
filteredWords = [word for word in metin.split() if word.lower() not in tr_stopwords]
print(" ".join(filteredWords))







