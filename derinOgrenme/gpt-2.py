"""
metin üretimi

gpt-2 metin üretimi

llama

"""

# import libraries 
# huging face kütüphanesinin NLP de çok kullanılır
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# modelin tanımlanması
model_name = "gpt2" # gpt3 şu an biziöm kullanımıma acık değil kütüphane olarak openAı apısinden kullanabilirsin
# farklı kütüphaneler denemekten eckinme zor geliyorsa feğiştirmeyide düşün

# tokenizer tanımlama ve model oluşturma
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)

# metin üretimi için gerekli olan başlangıç texti
text = input("metin giriniz")

# tokenization
inputs = tokenizer.encode(text, return_tensors="pt", padding=True)  # retorn tensor cıktının python tensoru olmasını sağlar

# metin üretimi gerçekleştirme
outputs = model.generate(
    inputs,    #modelin başlangıç noktası
    max_length=50,  # Üretilecek metnin maksimum uzunluğu
    num_return_sequences=1,  # Üretilecek metin sayısı
    no_repeat_ngram_size=2,  # Tekrar eden kelime gruplarını engelle
    temperature=0.7,  # Çeşitlilik (düşük değer = tutarlı, yüksek değer = yaratıcı)
    top_k=50,  # En olası 50 tahmini dikkate al
    top_p=0.9,  # Cümle olasılığını kontrol eden parametre
)


# modelin ürettiği tokenları okunabilir hale getirmek
generated_text = tokenizer.decode(outputs[0], skip_special_tokens  = True)  # modelin başlangıc ve bitiş tokeni var çümle bittimi başladımı bize göstermesin metinden cıkartsın ama kendisi bilsin
# ozel tokenları görmemek için 2. parametreyi verdik

# üretilen metin
print(generated_text)






