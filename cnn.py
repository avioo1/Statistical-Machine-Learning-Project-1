# CNN for Author Attribution
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
from keras.utils import to_categorical
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
import keras.optimizers
from keras.layers import Embedding
from collections import Counter
import operator 
from sklearn.model_selection import train_test_split

modelw2v = KeyedVectors.load_word2vec_format('./data/w2v_model_M.bin', binary = True)


embeddings_index = {}
for word in modelw2v.wv.vocab.keys():
    embeddings_index[word] = modelw2v.wv[word]


file_path = './data/train_tweets_mod.csv'
dfs = pd.read_csv(file_path)
num_entries = 0
users = dfs['User'].values.tolist()
user_tweet_dict = Counter(users)
sorted_user_tweet_dict = sorted(user_tweet_dict.items(), key=operator.itemgetter(1), reverse=True)
values = user_tweet_dict.values()
count_dict = Counter(values)
sorted_count_dict = sorted(count_dict.items(), key=operator.itemgetter(1), reverse=True)
valid_users = [user[0] for user in sorted_user_tweet_dict if user[1] > 39]
dfs = dfs[dfs['User'].isin(valid_users)]
keys = range(len(valid_users))
dict1 = dict(zip(keys, valid_users))
dict2 = dict(zip(valid_users, keys))
lst = []
for i in range(0, len(dfs)):
    lst.append(dict2[dfs.iloc[i]['User']])
dfs['Userd'] = lst
x = dfs['Clean Tweets']
y = dfs.Userd
x_train3, x_test3, y_train3, y_test3 = train_test_split(x, y, test_size=0.2, random_state = 42)


tokenizer = Tokenizer(num_words = 30000)
tokenizer.fit_on_texts(x_train3)
sequences = tokenizer.texts_to_sequences(x_train3)
x_train3_seq = pad_sequences(sequences, maxlen = 200)
sequences_test = tokenizer.texts_to_sequences(x_test3)
x_val_seq = pad_sequences(sequences_test, maxlen = 200)

num_words = 30000
embedding_matrix = np.zeros((num_words, 300))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


      
y_train3 = to_categorical(y_train3)
y_test3 = to_categorical(y_test3) 

cnn = Sequential()
e = Embedding(30000, 300, weights=[embedding_matrix], input_length = 200, trainable=True)
cnn.add(e)
cnn.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
cnn.add(GlobalMaxPooling1D())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dense(1931, activation='sigmoid'))
cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn.fit(x_train3_seq, y_train3, validation_data=(x_val_seq, y_test3), epochs=10, batch_size=32, verbose=2)
        

test_df = pd.read_csv("./data/test_tweets_mod.csv")
test_df = test_df.replace(np.nan, '', regex = True)
x1 = test_df['Clean Tweets']
seq_test = tokenizer.texts_to_sequences(x1)
x_test_seq = pad_sequences(sequences_test, maxlen = 200)
predictions = cnn.predict_classes(x_test_seq)

for i in range(0, len(predictions)):
    predictions[i] = dict1[predictions[i]]
test_df['predictions'] = predictions
test_df.to_csv("testing.csv")
