#!/usr/bin/env python

"""

This is a python script that uses keras to build an lstm network.

Training is done on a simplified dataset 'simplified_dataset.csv' located in the same folder with this file

The problem as is, mapping an input letter to an output number (class), is not very suitable for a 
sequence model as LSTM, but still the point of this brief example is to show:
- how to build an LSTM neural network with keras
- how to preprocess data in to One-hot vectors
- how to shape data in a suitable format for keras
- how to do predictions with the keras trained model.

"""

import numpy as np
import pandas as pd
import sys
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils.np_utils import to_categorical

# Change epochs here
EPOCHS = 1000

# Read data
df = pd.read_csv('simplified_dataset.csv')
X = df.iloc[:, 0]
y = df.iloc[:, 1]

# Convert input data X to one-hot vector
lex = LabelEncoder()
Xle = lex.fit_transform(X)
Xoh = np.array(pd.get_dummies(Xle))

# Convert output data y to one-hot vector
yoh = np.array(pd.get_dummies(y))

# Reshape data for the LSTM layer
Xoh = np.reshape(Xoh, (Xoh.shape[0], Xoh.shape[1], 1))

# Build a model
model = Sequential()
model.add(LSTM(units=10, input_shape=(Xoh.shape[1], 1), return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(units=10))
model.add(Dropout(0.1))
model.add(Dense(units=21))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
print(model.summary())

model.fit(x=Xoh, y=yoh, epochs=EPOCHS, batch_size=7)

# Evaluate model
scores = model.evaluate(Xoh, yoh)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# Get a sample from input and predict its output
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
    in_oh = Xoh[in_le[0], :, :]
    in_oh = np.reshape(in_oh, (1, in_oh.shape[0], 1))
    
    # Predict
    preds = model.predict(in_oh)
    print("Prediction:", np.argmax(preds)+1)




