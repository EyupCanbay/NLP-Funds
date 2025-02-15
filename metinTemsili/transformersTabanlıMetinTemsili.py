# kütüphaneleri import et
from transformers import AutoTokenizer, AutoModel
import torch
import numpy

# model ve tokenizer yükle
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# input ve test metni tanımla
text = "Transformers can be used for natural language processing"


# metni tokenlara dönüştür
inputs = tokenizer(text, return_tensors="pt") # tokanları teron ederken pytorch pt formatında retrun etsin istiyoruz

# modeli kullanarak metin temsili oluştur
with torch.no_grad():      # gradyanların hesaplanmasını durdurur böylece belleği daha verimli kullanırız
    outputs = model(**inputs)


#modelin çıkışından son gizli durumu alalim
last_hidden_state = outputs.last_hidden_state # tüm token cıktılarını almak için

# ilk token embedingini alalım 
first_token_embedding = last_hidden_state[0,0,:].numpy() 

print("metin temsili", first_token_embedding)



