"""
classification
bir çümlenin olumlu veya olumsuz oldğunu sezme
sentement analiz gerçekleşticem
"""
# import library

import nltk
from nltk.classify import MaxentClassifier

# veri seti tanımlama
train_data = [
    ({"love":True, "amazing":True, "happy":True, "terrible":False}, "pozitive"),
    ({"hate":True, "happy":False, "terrible":True}, "negative"),
    ({"joy": True, "happy":True, "hate":False}, "positive"),
    ({"sad": True, "depressed":True, "love":False}, "negative"),
]

# maximum entropi classifier train etme
classifier = MaxentClassifier.train(train_data, max_iter = 10)
print(classifier)

# yeni çümleyle test

test_sentence= "I hate movie and it was terrible"
features = {word: (word in test_sentence.lower().split()) for word in ["love", "amazing", "terrible", "happy", "joy", "sad"]}

label= classifier.classify(features)
print("Result",label)






