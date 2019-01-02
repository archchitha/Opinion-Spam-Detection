import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


df_data = pd.read_csv('data/deceptive-opinion.csv', sep=',', names=['label', 'hotel', 'polarity', 'source', 'review_text'])
from sklearn.model_selection import train_test_split
text = df_data['review_text'].values
y = df_data['label'].values
text_train, text_test, y_train, y_test = train_test_split(text, y, test_size=0.25, random_state=1000)



from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(text_train)

x_train = vectorizer.transform(text_train)
x_test = vectorizer.transform(text_test)


from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(x_train, y_train)
score = classifier.score(x_test, y_test)

print("LogisticRegression Accuracy:", score)


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

score= accuracy_score(y_test, prediction)
print("NB Accuracy:", score)

from sklearn.svm import LinearSVC

clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

score= accuracy_score(y_test, prediction)
print("SVM Accuracy:", score)


from keras.models import Sequential
from keras import layers

input_dim = x_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


history = model.fit(x_train, y_train,
                    epochs=100,
                    verbose=False,
                    validation_data=(x_test, y_test),
                    batch_size=10)

loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print("ANN Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("ANN Testing Accuracy:  {:.4f}".format(accuracy))
