import json
import string
import random

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

data_file = open('chatbot-nlp/intents.json').read()
data = json.loads(data_file)

words = []
classes = []
data_X = []
data_Y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        data_X.append(pattern)
        data_Y.append(intent["tag"])

    if intent["tag"] not in classes:
        classes.append(intent["tag"])

lemmatizer = WordNetLemmatizer()

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words = sorted(set(words))

classes = sorted(set(classes))

training = []
out_empty = [0] * len(classes)

for idx, doc, in enumerate(data_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)

    output_row = list(out_empty)
    output_row[classes.index(data_Y[idx])] = 1
    training.append([bow, output_row])
random.shuffle(training)
training = np.array(training, dtype=object)
train_X = np.array(list(training[:,0]))
train_Y = np.array(list(training[:,1]))

model = Sequential()
model.add(Dense(128, input_shape = (len(train_X[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_Y[0]), activation="softmax"))
model.compile(loss = 'categorical_crossentropy',
              optimizer ='adam',
              metrics = ["accuracy"]
              )

print(model.summary())
model.fit(x=train_X, y=train_Y, epochs=150, verbose = 1)

