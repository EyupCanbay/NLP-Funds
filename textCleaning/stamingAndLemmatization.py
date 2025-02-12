import nltk

nltk.download("wordnet")  # lemmatization için gerekli veri tabanı

from nltk.stem import PorterStemmer # stamming için fonksiyon

#porter stemmer nesnesini oluşturalım
stemmer = PorterStemmer()

words = ["write", "wrote", "written", "wroter","better","went","came"]
print(words)
#kelimelerin stemlarını buluyoruz, bunu yaparken porter stemerin stem() fonk kullanıları
stems = [stemmer.stem(w) for w in words]
print(stems)

# lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

lemmas = [lemmatizer.lemmatize(w, pos="v") for w in words] # ikinci bir parametre verdik çünü en anlamı en kücük köklerii istiyoruz ve fiil olarak işlemesini istedik o yüzden filleri belirtittik
print(lemmas)   # verblerin anlamlı köklerini bulmasını istiyoruz ve buluyor pos="v" demedmizin sebebi odur


