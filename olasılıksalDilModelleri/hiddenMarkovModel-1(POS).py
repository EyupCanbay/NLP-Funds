"""
 part of speach iin kullanıcam ebn
 pekilemerin uygun sözcük türünü bulma çalışması 

i(zamir) am a teacher(noun)
"""

# import library
import nltk
from nltk.tag import hmm

# ornek traning Data tanımla
train_data = [
    [("I", "PRP"), ("am", "VBP"), ("a", "DT"), ("teacher", "NN")],
    [("You", "PRP"), ("are", "VBP"), ("a", "DT"), ("student", "NN")],
    [("He", "PRP"), ("is", "VBP"), ("an", "DT"), ("engineer", "NN")],
    [("She", "PRP"), ("was", "VBD"), ("a", "DT"), ("doctor", "NN")],
    [("They", "PRP"), ("were", "VBD"), ("some", "DT"), ("artists", "NNS")]
]

# train HMM
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)
print(hmm_tagger)
# yeni bir çümle oluştur içinde bulunan her bir şözcüğün türünü etiketle



test_sentence = "I am a student".split()

tags = hmm_tagger.tag(test_sentence)
print(tags)














