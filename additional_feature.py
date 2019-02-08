import numpy as np
import pandas as pd
from collections import defaultdict
import re
import textblob
import sys
import os
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras import regularizers
from keras.layers import LSTM,concatenate, Bidirectional, Dropout, Reshape


MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

texts = []
labels = []


df_data = pd.read_csv('data/deceptive-opinion.csv', sep=',', names=['label', 'hotel', 'polarity', 'source', 'review_text'])
from sklearn.model_selection import train_test_split

df_data['no_of_words'] = df_data['review_text'].apply(lambda x: len(str(x).split(" ")))

df_data['no_of_characters'] = df_data['review_text'].str.len() ## this also includes spaces

from nltk.corpus import stopwords
stop = stopwords.words('english')

df_data['stopwords'] = df_data['review_text'].apply(lambda x: len([x for x in x.split() if x in stop]))

df_data['hastags'] = df_data['review_text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))

df_data['numerics'] = df_data['review_text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

df_data['upper'] = df_data['review_text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

df_data['noun_count'] = df_data['review_text'].apply(lambda x: check_pos_tag(x, 'noun'))
df_data['verb_count'] = df_data['review_text'].apply(lambda x: check_pos_tag(x, 'verb'))
df_data['adj_count'] = df_data['review_text'].apply(lambda x: check_pos_tag(x, 'adj'))
df_data['adv_count'] = df_data['review_text'].apply(lambda x: check_pos_tag(x, 'adv'))
df_data['pron_count'] = df_data['review_text'].apply(lambda x: check_pos_tag(x, 'pron'))

texts = df_data['review_text'].values
labels = df_data['label'].values

pol = df_data['no_of_words'].values



def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()



from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
pol= to_categorical(np.asarray(pol))

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
pol=pol[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
x1_train=pol[:-nb_validation_samples:]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
x1_val =pol[-nb_validation_samples:]

print('Number of positive and negative reviews in traing and validation set ')
print
y_train.sum(axis=0)
print
y_val.sum(axis=0)

GLOVE_DIR = "/ext/home/analyst/Testground/data/glove"
embeddings_index = {}
f = open('data/wiki-news-300d-1M.vec')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
pol = Input(shape=(785,), name='meta_input')


embedded_sequences = embedding_layer(sequence_input)
l_cov1 = Conv1D(128, 5, activation='relu')(embedded_sequences)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(5)(l_cov2)
l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
l_pool3 = MaxPooling1D(35)(l_cov3)
l_flat = Flatten()(l_pool3)
l_dense = Dense(10, activation='relu')(l_flat)
preds = Dense(2, activation='softmax')(l_dense)

x = concatenate([l_dense, pol], axis=-1)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(10, activation='sigmoid')(x)
main_output = Dense(2, activation='sigmoid', name='main_output')(x)
model = Model(inputs=[sequence_input, pol], outputs=[main_output])


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'], loss_weights=[1.])

print("model fitting - simplified convolutional neural network")
model.summary()

print('Shape of X train and X validation tensor:', x_train.shape, x_val.shape)
print('Shape of X1 train and X1 validation tensor:', x1_train.shape, x1_val.shape)
print('Shape of label train and validation tensor:', y_train.shape, y_val.shape)



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

history = model.fit([x_train, x1_train], y_train,
                    epochs=20,
                    verbose=1,
                    validation_data=([x_val, x1_val], y_val),
                    batch_size=10)
loss, accuracy = model.evaluate([x_train, x1_train], y_train, verbose=1)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate([x_val, x1_val], y_val, verbose=1)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)

"""
from keras.utils import plot_model
plot_model(model, to_file='model.png')
"""