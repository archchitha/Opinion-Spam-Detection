import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import nltk
nltk.download('stopwords')
import re



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
review_train, review_test, y_train, y_test = train_test_split(text, y, test_size=0.25, random_state=1000)


from keras.models import Sequential
from keras import layers

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV


"""
df_data = pd.read_csv('data/amazon.txt', delimiter = "\t", names=['DOC_ID',"label",'RATING','VERIFIED_PURCHASE','PRODUCT_CATEGORY','PRODUCT_ID','PRODUCT_TITLE','REVIEW_TITLE','review_text'])

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le = preprocessing.LabelEncoder()
le.fit(["label"])

from sklearn.model_selection import train_test_split
text = df_data['review_text'].values
y = le.fit_transform(df_data['label'].values)
review_train, review_test, y_train, y_test = train_test_split(text, y, test_size=0.25, random_state=1000)
"""


from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(review_train)
X_train = tokenizer.texts_to_sequences(review_train)
X_test = tokenizer.texts_to_sequences(review_test)
vocab_size = len(tokenizer.word_index) + 1

from keras.preprocessing.sequence import pad_sequences

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)



def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model



epochs = 20
embedding_dim = 50
output_file = 'data/output.txt'


param_grid = dict(num_filters=[32, 64, 128],
                      kernel_size=[3, 5, 7],
                      vocab_size=[vocab_size],
                      embedding_dim=[embedding_dim],
                      maxlen=[maxlen])
model = KerasClassifier(build_fn=create_model,
                            epochs=epochs, batch_size=100,
                            verbose=False)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                              cv=4, verbose=1, n_iter=5)
grid_result = grid.fit(X_train, y_train)

test_accuracy = grid.score(X_test, y_test)


prompt = input(f'finished {df_data}; write to file and proceed? [y/n]')

with open(output_file, 'a') as f:
    s = ('Best Accuracy : '
            '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
    output_string = s.format(
            grid_result.best_score_,
            grid_result.best_params_,
            test_accuracy)
    print(output_string)
    f.write(output_string)


#number of filters, the kernel size, and the activation function.

"""
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

history = model.fit(X_train, y_train,
                    epochs=10,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)
"""
