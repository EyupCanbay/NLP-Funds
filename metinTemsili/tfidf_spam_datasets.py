#import libraries
# upload datasets
#tfidf
#examine word set
# create a dataframe to include tfidf scors
# sort the scores and examine

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("./datasets/spam.csv",encoding="latin1")

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df.v2)
#print(x)
feature_names = vectorizer.get_feature_names_out()
print(feature_names)

tfidfScore = x.mean(axis=0).A1  # herkelimenin ortaama tfidf değerleri


df_tfidf = pd.DataFrame({"word":feature_names, "tfidf_score": tfidfScore})
print(df_tfidf)

sorted_tf_idf = df_tfidf.sort_values(by="tfidf_score",ascending=False)

# Sonucu göster
print(sorted_tf_idf)