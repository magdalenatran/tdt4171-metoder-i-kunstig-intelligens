import pickle
from os import path
import tensorflow as tf
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import RMSprop

# finds the file and opens it
file = path.abspath("keras-data.pickle")
data = pickle.load(open(file,"rb"))

# initialise
x_train, y_train, x_test, y_test = data["x_train"], data["y_train"], data["x_test"], data["y_test"]

# holds information about the number of different word in the vocabulary
vocab_size = data["vocab_size"]

# gives the longest review in the dataset in terms of the number of words used
max_length = data["max_length"]

x_train = pad_sequences(x_train, max_length)
x_test = pad_sequences(x_test, max_length)

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# builds the model
model = tf.keras.Sequential()

# embeds the word-vectors into a high-him metric space
model.add(tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = 256))

# adds a LSTM layer and a Dense layer at the end
model.add(LSTM(256, activation ="relu"))
model.add(Dense(2, activation = "sigmoid"))

# compiles the model
model.compile(optimizer=RMSprop(), loss="binary_crossentropy", metrics=["accuracy"])

# trains the model when the model is done
model.fit(x_train, y_train, epochs=3, batch_size = 256)

model.save("models/lstm.h5")

# evalutes the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

print("The loss is:", loss)
print("The accuracy is:", accuracy)