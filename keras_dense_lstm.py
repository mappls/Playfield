#!/usr/bin/env python

"""

This is a python script that uses keras to build an lstm network.

Training is done on a simplified dataset provided by James:
https://docs.google.com/spreadsheets/d/1BfBgQw9UOphlq4RfzY9xC7k4j7gggm7bORpMWt74OOY/edit#gid=0

"""

import numpy as np
import pandas as pd
import sys


from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils.np_utils import to_categorical


# Read data
df = pd.read_csv('simplified_dataset.csv')
print(df.head())
X = df.iloc[:, 0]
y = df.iloc[:, 1]


# Convert input data X to one-hot vector
lex = LabelEncoder()
Xle = lex.fit_transform(X)
Xoh = pd.get_dummies(Xle)

# Convert output data y to one-hot vector
yoh = pd.get_dummies(y)
print("yoh.shape:", yoh)

# Build a model
model = Sequential()
# model.add(LSTM(units=10, input_dim=input_len, return_sequences=True))
model.add(Dense(units=10, input_dim=21, activation='relu'))
model.add(Dense(units=21, activation='softmax'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
print(model.summary())

model.fit(x=Xoh, y=yoh, epochs=300, batch_size=7)

scores = model.evaluate(Xoh, yoh)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# Get a random sample of X and predict its output
index = np.random.randint(0,len(X))
sampleX = np.reshape(Xoh[index], (1, 21))
print("Input:", X.iloc[np.argmax(sampleX)])
preds = model.predict(sampleX)
print("predicted output:", np.argmax(preds)+1)

