"""
solve classification problem (sentiment analysis in NLP ) with RNN (Deep Learning Based Languaga Model)
PROBLEM -> duygu analizi
Restaurant giden müşterilerin yorumlarının olumlumu olumsuz mu olduğunu değerlendirilmesi
"""
# import library
import pandas as pd
import numpy as np

from gensim.models import Word2Vec #metin temsili

#FOR RNN 
from tensorflow.keras.preprocessing.sequence import pad_sequences   # pedding için
from tensorflow.keras.models import Sequential   # kerasın içindeki layerlerşn RNN i inşa etmek için bir base oturması gerekiyor buda sequantial olucak yani layerleri sequentaln içine eklicez
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding   # katmanlar
from tensorflow.keras.preprocessing.text import Tokenizer    # tokizasyon için

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder   # etiketleri ekod etmek için kullanıcaz
# create datasets 
#veriseti üreticisi lazım CHATGPTden küçük veri seti oluşturcam
data = {
    "text": [
        "Yemekler harikaydı, servis çok hızlıydı.",
        "Mükemmel bir deneyimdi, kesinlikle tekrar geleceğim!",
        "Tatlılar çok taze ve lezzetliydi.",
        "Personel çok kibar ve yardımseverdi.",
        "Fiyat performans açısından çok başarılıydı.",
        "Sunum çok şıktı ve yemekler çok lezzetliydi.",
        "Garsonlar güler yüzlü ve ilgiliydi.",
        "Mekan çok şık ve atmosferi harikaydı.",
        "Et mükemmel pişmişti, tam kıvamındaydı.",
        "Fiyatlar çok uygundu, lezzet de çok iyiydi.",
        "Çalışanlar çok nazik ve yardımseverdi.",
        "Hızlı servis, güzel yemekler, harika deneyim!",
        "Porsiyonlar çok büyük ve doyurucuydu.",
        "Şefin özel yemeği gerçekten muhteşemdi.",
        "Manzara harikaydı, yemekler de çok lezzetliydi.",
        "Çocuklar için çok güzel bir alan vardı.",
        "Müşteri memnuniyeti için harika bir yer.",
        "Lezzet olarak mükemmeldi, kesinlikle tavsiye ederim!",
        "Ev yemekleri gibiydi, çok sıcak ve lezzetliydi.",
        "Baharat dengesi harikaydı, tam kıvamında pişmişti.",
        "Yemekler tam zamanında geldi, sıcak ve lezzetliydi.",
        "Özel günler için mükemmel bir mekan!",
        "Her şey çok temiz ve düzenliydi.",
        "Tatlılar efsaneydi, özellikle cheesecake harikaydı!",
        "Harika bir akşam yemeğiydi, her şey mükemmeldi!",
        "Garson çok ilgisizdi ve yemekler soğuktu.",
        "Yemeklerin lezzeti beklentimin altındaydı.",
        "Ortam çok gürültülüydü, yemek keyfi yapamadım.",
        "Servis çok yavaştı, çok uzun süre bekledik.",
        "Hijyen konusunda ciddi eksiklikleri vardı.",
        "Menü çok sınırlıydı, seçenekler yetersizdi.",
        "Siparişim yanlış geldi ve çok bekledim.",
        "Yemekler fazla tuzluydu ve ağırdı.",
        "Servis çok kötüydü, ilgilenen kimse yoktu.",
        "Yemekler aşırı yağlıydı, mideyi rahatsız etti.",
        "Tatlı çok bayattı ve lezzetsizdi.",
        "Beklentimin çok altında bir hizmetti.",
        "Sipariş verdikten sonra çok uzun bekledik.",
        "Masalar çok kirliydi ve kötü kokuyordu.",
        "Kredi kartı geçmiyordu, çok saçma bir durumdu.",
        "Yemekler eksik geldi, servis yetersizdi.",
        "Fiyatlar aşırı pahalı ve porsiyonlar küçüktü.",
        "Garson siparişleri sürekli karıştırdı.",
        "Müzik sesi çok yüksekti, rahatsız ediciydi.",
        "Masalar arasında çok fazla mesafe yoktu, sıkışıktı.",
        "Çalışanlar ilgisizdi, servis çok gecikti.",
        "Tavuk çok kuru ve tatsızdı.",
        "Yemekler çok yağlıydı ve ağırdı.",
        "Kötü bir deneyimdi, tekrar gitmem.",
        "Siparişim yanlış geldi, çok beklemek zorunda kaldım.",
        "Çalışanlar ilgisizdi ve sipariş almak çok uzun sürdü.",
        "Yemekler çok soğuk geldi, hiç keyif alamadım.",
        "Restorandaki temizlik yetersizdi, hijyen sorunu vardı.",
        "Garsonlar çok suratsızdı ve ilgisizdi.",
        "Menüde çok az seçenek vardı, çeşitlilik yetersizdi.",
        "Ortam çok gürültülüydü, sohbet bile edemedik.",
        "Yemeklerin porsiyonları çok küçüktü, doymadık.",
        "Servis yavaş ve düzensizdi, siparişler yanlış geldi.",
        "Fiyatlar çok yüksekti, ödediğimiz paraya değmedi.",
        "Tatlı çok şekerliydi ve bayat tadı vardı.",
        "Masalar çok sıkışıktı, rahat oturamadık.",
        "İçecekler çok pahalıydı ve tadı kötüydü.",
        "Garson siparişi yanlış aldı ve düzeltmek için uğraşmadı.",
        "Beklentilerimizi hiç karşılamadı, büyük hayal kırıklığıydı.",
        "Pişme derecesi iyi ayarlanmamıştı, et çok sertti.",
        "Müzik sesi çok yüksekti ve rahatsız ediciydi.",
        "Salata bayattı ve lezzetsizdi.",
        "Servis çok yavaştı, çok uzun süre bekledik.",
        "Porsiyonlar küçüktü ve yemek lezzetsizdi.",
        "Masalar kirliydi ve düzgün temizlenmemişti.",
        "Hizmet çok kötüydü, müşteriyle ilgilenilmiyordu.",
        "Çok kötü bir deneyimdi, tekrar gelmeyi düşünmüyoruz.",
        "Lezzet inanılmazdı, porsiyonlar çok doyurucuydu.",
        "Restoranın atmosferi çok sıcak ve huzurluydu.",
        "Garsonlar çok kibar ve yardımseverdi.",
        "Ana yemekler harikaydı, özellikle et yemeği mükemmeldi.",
        "Tatlılar tam kıvamındaydı, çok beğendik.",
        "Servis hızı mükemmeldi, bekletilmeden servis edildi.",
        "Yemek sunumu çok şık ve özenliydi.",
        "Fiyatlar çok makul ve lezzet mükemmeldi.",
        "Menüde her damak zevkine hitap eden seçenekler vardı.",
        "Kahvaltı tabağı çok zengin ve tazeydi.",
        "Restoranın dekorasyonu çok hoş ve moderndi.",
        "Yemeklerin baharat dengesi çok iyi ayarlanmıştı.",
        "Açık hava oturma alanı çok keyifliydi.",
        "Deniz ürünleri taze ve lezzetliydi.",
        "Garsonlar hızlı ve dikkatliydi, siparişler eksiksiz geldi.",
        "Mekan çok ferah ve temizdi.",
        "Yemekler tam zamanında ve sıcak servis edildi.",
        "Kendi spesiyalleri çok lezzetliydi, mutlaka denemelisiniz!",
        "Çalışanlar güler yüzlü ve ilgiliydi.",
        "Müşteri memnuniyeti ön planda tutuluyordu.",
        "Kokteyller harikaydı, çok beğendik!",
        "Şefin önerdiği yemekler tam bizim damak zevkimize uygundu.",
        "Yemeklerin yanında gelen soslar çok lezzetliydi.",
        "Aile yemekleri için mükemmel bir mekan."
    ],
    "label": [
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "positive", "positive", 
    "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive"
]
}

df = pd.DataFrame(data)
print(df)
# metin temizleme ve prepcessing: tokenization, padding, label engoding, train test split

# tokenization 
tokenizer = Tokenizer()  
tokenizer.fit_on_texts(df["text"])  
sequences = tokenizer.texts_to_sequences(df["text"])
word_index = tokenizer.word_index


# pedding -> df ininde farklı çümleleri fixlemek için yapıyom
maxlen = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen = maxlen)
print (X.shape)

#label encoding
labeL_encoder = LabelEncoder()
Y = labeL_encoder.fit_transform(df["label"])

#tain test split
x_train, x_text, y_train, y_text = train_test_split(X,Y, test_size= 0.2, random_state= 42)




# metin temsili: word embeding : Word2Vec
sentences = [text.split() for text in df["text"]]
word2vec_model = Word2Vec(sentences, vector_size=50, window=5, min_count=1)   

embedding_dim = 50
embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]
 





# modelling:build train ve test  RNN modeli  

#build model
model = Sequential()

# embedding 
model.add(Embedding(input_dim= len(word_index) + 1, output_dim = embedding_dim, weights=[embedding_matrix], input_length = maxlen, trainable = False))

#rnn layer
model.add(SimpleRNN(50, return_sequences=False)) 

#output layer
model.add(Dense(1,activation="sigmoid"))

#COMPİLE MODEL
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#train model
model.fit(x_train, y_train, epochs=17, batch_size=2, validation_data=(x_text, y_text))

#evaluate rnn model
test_loss, test_accurary = model.evaluate(x_text,  y_text)
print("test loss:", test_loss, "test accurary:" , test_accurary)




# çümle sınıflandırma çalışması
def classify_sentence(sentence):
    seq = tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(seq, maxlen = maxlen)
    prediction = model.predict(padded_seq)

    predicted_calss = (prediction > 0.5).astype(int)
    label = "positive" if predicted_calss[0][0] == 1 else "negative"
    print(prediction)
    return label




sentence = "aşırı güzeldi kesinlikle."

result = classify_sentence(sentence)
print("Result:" , result)



