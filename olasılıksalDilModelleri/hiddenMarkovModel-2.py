# import libraryies 
import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000 #pos tag işlemi için sık kullanılır 

# gelerkli verisetini içeri aktar
#nltk.download("conll2000")

train = conll2000.tagged_sents("train.txt")
test = conll2000.tagged_sents("test.txt")
#print(train[:1])
# train hmm
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train)


# yeni çümle ve test
test_sentence = "I like going to school".split()
tags = hmm_tagger.tag(test_sentence)
print(tags)




