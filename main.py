import re
import nltk
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def clean_tweets(text, stem=False):
    text = re.sub(r'RT[\s]+', '', text)  # remove RETWEET: RT
    text = re.sub(r'http\S+', '', text)  # remove links
    text = re.sub(r'(c\\x(.){2})', '', text)  # remove encoded text
    text = re.sub(r'(\\x(.){2})', '', text)  # remove encoded text
    text = re.sub("@S+|https?:S+|http?:S|[^A-Za-z0-9]+", ' ', str(text).lower()).strip()
    text = text[2:]

    # store token into empty list:
    tokens = []
    for token in text.split():
        # token is a stop word?
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)


def extract_hash_tags(text):
    hashtags = []
    for token in text.split():
        hashtags.append(token)
    return hashtags


def predictCongressionalParty(score):
    aArray = []
    if score > 0.5:
        aArray.append(1)
    else:
        aArray.append(0)
    return aArray


nltk.download('stopwords')
stemmer = SnowballStemmer('english')
stop_words = stopwords.words('english')
# print(stop_words)

# ------------------------------ Part 1 ------------------------------ #
# Get Data
Original_df = pd.read_csv("congressional_tweet_training_data.csv/congressional_tweet_training_data.csv")
Original_df = Original_df.dropna()  # Drop missing info row
Original_df = Original_df.drop_duplicates()  # remove duplicate


# print(df['hashtags'].value_counts())
# Extract hashtags
Original_df['commonHashtags'] = Original_df.groupby('hashtags')['hashtags'].transform('count')

# Sort by commonHashtags
Original_df = Original_df.sort_values('commonHashtags', ascending=False)
# print(Original_df.head(500))
# Clean tweet
Original_df.full_text = Original_df.full_text.apply(lambda x: clean_tweets(x))
# print(df.full_text)

# Labels 1 0 for target parties
Original_df["party_id"] = [1 if each == "D" else 0 for each in Original_df.party_id]

# Normalize Columns: favorite_count and retweet_count.
cols_to_norm = ['favorite_count', 'retweet_count']
Original_df[cols_to_norm] = Original_df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Create new dataFrame
df = Original_df.copy()

# Choose 500 most common
df = df.head(500)

# Drop unnecessary columns
df = df.drop(['hashtags', 'commonHashtags', 'year'], axis=1)

# ------------------------------ Part 2 ------------------------------ #
# SPLITTING DATA 80/20
train_data, test_data = train_test_split(df, test_size=0.2, random_state=16)
# print("Train Data size:", len(train_data))
# print("Test Data size", len(test_data))

tokenizer = Tokenizer()
# train_data.full_text is a list of string.
# fit_on_text: update vocabulary by train_data.full_text
tokenizer.fit_on_texts(train_data.full_text)
word_index = tokenizer.word_index

# print(word_index)
vocab_size = len(tokenizer.word_index) + 1
# print("Vocabulary Size :", vocab_size)

handfulLengthTweet = 50

# The tokens are converted into sequences and then passed to the pad_sequences() function
x_train = pad_sequences(tokenizer.texts_to_sequences(train_data.full_text), maxlen=handfulLengthTweet)
x_test = pad_sequences(tokenizer.texts_to_sequences(test_data.full_text), maxlen=handfulLengthTweet)

# Encode label
encoder = LabelEncoder()
encoder.fit(train_data.party_id.to_list())
y_train = encoder.transform(train_data.party_id.to_list())
y_test = encoder.transform(test_data.party_id.to_list())
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

embeddings_index = {}
# Download Glove embeddings file
f = open('glove.6B.300d.txt', encoding="utf8")
for line in f:
    # For each line file, the words are split and stored in a list
    values = line.split()
    word = value = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
# print('%s word vectors.' % len(embeddings_index))

# Embedding matrix with zeroes of shape vocab x embedding dimension
embedding_matrix = np.zeros((vocab_size, 300))
# Iterate through word, index in the dictionary
for word, i in word_index.items():
    # extract the corresponding vector for the vocab indices of same word
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Storing it in a matrix
        embedding_matrix[i] = embedding_vector



embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix],
                            input_length=handfulLengthTweet, trainable=False)


# ------------------------------ BASE MODEL ------------------------------ #
basemodel = Sequential()
basemodel.add(embedding_layer)
basemodel.add(Dense(300, activation='relu', input_dim=300))
basemodel.add(Dense(128, activation='relu'))
basemodel.add(Dense(64, activation='relu'))
basemodel.add(Dense(1, activation='sigmoid'))
basemodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = basemodel.fit(x_train, y_train, batch_size=5,
                    epochs=20, validation_data=(x_test, y_test))
print('Accuracy: %f' % basemodel.evaluate(x_test, y_test)[1])

# ------------------------------ MODEL ------------------------------ #
# a Keras model
model = Sequential()
model.add(embedding_layer)
# model_II.add(keras.layers.Dense(12, input_dim=handfulLengthTweet, activation='relu'))
# model_II.add(keras.layers.Dense(8, activation='relu'))
# model_II.add(keras.layers.Dense(1, activation='sigmoid'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model
print("Training Model:")
training = model.fit(x_train, y_train,
                     batch_size=5, epochs=20,
                     validation_data=(x_test, y_test))
print("Predict model:")
predictions = (model.predict(x_test) > 0.5).astype(int)
for i in range(int(len(y_test)/2)):
    print('%d (expected %d)' % (predictions[i], y_test[i]))
print('Accuracy: %f' % model.evaluate(x_test, y_test)[1])


