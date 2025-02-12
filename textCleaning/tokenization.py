import nltk # natural language toolkit 

nltk.download()  # metni kelime ve cümle bazında tokenlara ayırabilmek için gerekli
text = "Hello, World! How are you? Hello, hi..."

# Kelime Tokenizasyonu: word_tokenize: metni kelimelere ayırır, noktalama işaretleri 
# ve boşluklar ayrı birer token olarak lde edilecektir.
word_tokens = nltk.word_tokenize(text)

print(word_tokens)



# cümle tokenizasyonu: sent_tokenize: metni çümlelere ayırır
# her bir çümle birer token olur.


