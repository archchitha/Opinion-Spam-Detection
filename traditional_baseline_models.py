import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import nltk
"""nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')"""
import re

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string

def clean_review(text):
    # Strip HTML tags
    text = re.sub('<[^<]+?>', ' ', text)

    # Strip escaped quotes
    text = text.replace('\\"', '')

    # Strip quotes
    text = text.replace('"', '')

    return text


df_data = pd.read_csv('data/deceptive-opinion.csv', sep=',', names=['label', 'hotel', 'polarity', 'source', 'review_text'])
from sklearn.model_selection import train_test_split
text = df_data['review_text'].apply(clean_review)
y = df_data['label'].values
text_train, text_test, y_train, y_test = train_test_split(text, y, test_size=0.25, random_state=1000)


# create a count vectorizer object
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'),
                             lowercase=True, min_df=3, max_df=0.9,ngram_range=(2,3), max_features=5000)

vectorizer.fit(text)
x_train = vectorizer.transform(text_train)
x_test = vectorizer.transform(text_test)


"""
# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words=stopwords.words('english'),
                             lowercase=True, min_df=3, max_df=0.9, ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(text_train)
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(text_train)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(text_test)


# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word',stop_words=stopwords.words('english'),
                             lowercase=True, min_df=3, max_df=0.9, token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(text_train)
xtrain_tfidf =  tfidf_vect.transform(text_train)
xvalid_tfidf =  tfidf_vect.transform(text_test)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char',stop_words=stopwords.words('english'),
                             lowercase=True, min_df=3, max_df=0.9, token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(text_train)
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(text_train)
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(text_test)



"""
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

score= accuracy_score(y_test, prediction)
print("NB Accuracy:", score)



from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(x_train, y_train)
score = classifier.score(x_test, y_test)

print("LogisticRegression Accuracy:", score)



from sklearn.svm import LinearSVC

clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

score= accuracy_score(y_test, prediction)
print("SVM Accuracy:", score)
