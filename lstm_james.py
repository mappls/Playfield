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
X = df.iloc[:, 0]
y = df.iloc[:, 1]


# Convert input data X to one-hot vector
lex = LabelEncoder()
Xle = lex.fit_transform(X)
Xoh = pd.get_dummies(Xle)

# Convert output data y to one-hot vector
yoh = pd.get_dummies(y)

# Build a model
model = Sequential()
model.add(LSTM(units=10, input_dim=21, return_sequences=True))
# model.add(Dense(units=10, input_dim=21, activation='relu'))
model.add(Dense(units=21, activation='softmax'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
print(model.summary())

model.fit(x=Xoh, y=yoh, epochs=300, batch_size=7)

scores = model.evaluate(Xoh, yoh)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# Get a random sample of X and predict its output
while True:
    print("Please enter your input:")
    in_ = input()
    
    if in_ == 'exit':
        print('bye!')
        break
    
    if in_ not in X.tolist():
        print('Input not in training set!')
        continue
    
    # Encode input
    in_le = lex.transform([in_])
    in_oh = np.reshape(Xoh.iloc[in_le[0], :], (1,21))
    
    preds = model.predict(in_oh)
    print("Prediction:", np.argmax(preds)+1)




