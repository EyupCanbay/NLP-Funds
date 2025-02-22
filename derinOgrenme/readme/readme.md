RNN 
    RNN ile NLP alanında yapılan uygulamalar
        Dil Modellemen
        Makine Çevirisi
        Duygu Analizi
        Konuşma Tanıma
        Metin Üretme
    RNN nelerdegradiant sorunu var cok derin işlemlerde önceki veriyi kaybediyor çünkü zaman bağlı sürekli türev alınınca 0 a yaklaşıyor buda geçmişi unutmasına sebep oluyor ile
    RNN araştır
    TEKRARLAYAN SİNİR AĞIDIR
    

LSTM
    kullanım alanları
        doğal diş işlemlerde  ( NLP )
        konuşmayı yazıya dönüştürme   (classification)
        zaman dizisi verilerinden gelecekteki verileri tahmin etme       (prediction)
        müzik ve maetin üretimi işlemlerinde kullanılabilir
        videodaki kareler arasındaki zamansal bağımlılıklarıöğrenerek özet fealn cıkartabiliryoralr
        
        LSTM VE RNN ARASINDAKİ BENZERLİKLER FARKLILIKLAR
        ![alt text](image.png)





TRANSFORMERS
    BERT 
        dil anlama ve için kullanılır
        bert metnii hem geçmişine hem geleceğine bakarak bağlamı anlamaya çalışır
        metni anlamak için kullanılır örneğin soru cevaplama ve metni sınıflandırma

            transformers mimarisinin sadece encoder kısmını kullanılır
            transformer dikkat mekanızmasına dayanan bir modeldir
            iki aşamalı eğitim
                ön eğitim (pre-training)   
                    masked language modelling (MLM)  modelin rasgele maskelenmiş kelimeleri tahmin etmesi beklenir
                    next sentence prediction(NSP)   bir çümlenin mantıksal olarak uygun olup olmaması ile ilgilenir
                fine tuning
                     önceden eğitilmiş model sınıflandırma anlamsal ilişki çözme soru cevap gibi gibi şeyler için ince ayar yapılabilir
            çift yönlürdür
            transfer öğrenme
            transformer encoder kullanımı sadece burayı kullanılır anlama ve sınıflandırma konularında güzel kullanılır

    GPT
        metin üretme ve dil modelleme için kullanılan dilmodelidir
        metni soldan sağa olur bir kelimeyi tahmin etmek için önceki kelimelere dayanır
        metin üretme öykü yazma yaratıcı içerik üretmek için kullanırız
        TRANSFORMER MİAMİRİSİN DECODER KISMINI KULLANILIR
        otokorelasyonlu modeldir bu şekilde yaklaşır bu sayede mantıksal metinler ütermede olanak sağler
        tek aşamalı eğitime sahip

        özeliikleri
            tekyinlüdür
            metin üretimi yapabilir
            hikaye yazma sohbet botları yazar 
            transfer öğrenime küçük veri setleri ile güzel modeller oluşturabilirz
            cok büyük modeller vardır cpt3 175 milyar paremetrelidir




Özellik	BERT	GPT	LLaMA
Eğitim Yönü	Çift yönlü (bidir ectional)	Tek yönlü (unidirectional)	Çift yönlü
Kullanılan Transformer	Encoder	Decoder	Encoder + Decoder
Ana Görev	Metin anlama ve sınıflandırma	Metin üretimi ve dil modelleme	Hem metin üretimi hem metin anlama
Eğitim Görevleri	Masked Language Modeling, NSP	Language Modeling	Language Modeling
Kullanım Alanları	Soru-cevap, duygu analizi, NER	Metin üretimi, hikaye yazma, sohbet	NLP araştırmaları, düşük kaynaklı cihazlarda kullanım
Öne Çıkan Özellik	Çift yönlü bağlam öğrenme	Büyük metin üretimi, otokorelasyonlu	Verimlilik ve parametre açısından optimize edilmiş