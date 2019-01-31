import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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


df_data = pd.read_csv('data/deceptive-opinion.csv', sep=',', names=['label', 'hotel', 'polarity', 'source', 'review_text'])
from sklearn.model_selection import train_test_split
text = df_data['review_text'].values
y = df_data['label'].values
review_train, review_test, y_train, y_test = train_test_split(text, y, test_size=0.25, random_state=1000)


from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(review_train)
review_train = tokenizer.texts_to_sequences(review_train)
review_test = tokenizer.texts_to_sequences(review_test)
vocab_size = len(tokenizer.word_index) + 1

from keras.preprocessing.sequence import pad_sequences

maxlen = 100

X_train = pad_sequences(review_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(review_test, padding='post', maxlen=maxlen)



import numpy as np

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix


embedding_dim = 100
embedding_matrix = create_embedding_matrix(
    'data/glove.6B.100d.txt',
    tokenizer.word_index, embedding_dim)

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print(nonzero_elements / vocab_size)




from keras.models import Sequential
from keras import layers

embedding_dim = 100
model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim,
                           weights=[embedding_matrix],
                           input_length=maxlen,
                           trainable=True))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()



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
                    epochs=50,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)



