from keras.layers import LSTM, Dense, Activation, Dropout, Flatten
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle

def prepare_notes(notes):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(np.array(notes).reshape(-1,1))
    notes = list(scaler.transform(np.array(notes).reshape(-1,1)))
    # LSTM layers requires that data must have a certain shape
    # create list of lists fist
    notes = [list(note) for note in notes]

    # subsample data for training and prediction
    X = []
    y = []
    # number of notes in a batch
    n_prev = 30
    for i in range(len(notes)-n_prev):
        X.append(notes[i:i+n_prev])
        y.append(notes[i+n_prev])
    return X, y, scaler

# Modify conveniently with the path for your data
data_train_path = '../data/notes_train2'
with open(data_train_path, 'rb') as f:
    notes_train = pickle.load(f)
data_test_path = '../data/notes_test2'
with open(data_test_path, 'rb') as f:
    notes_test = pickle.load(f)

x_train, y_train, scaler = prepare_notes(notes_train)
x_test, y_test, _ = prepare_notes(notes_test)


n_prev = 30
model = Sequential()
model.add(LSTM(128, input_shape=(n_prev, 1), return_sequences=True))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(1))
model.add(Activation('linear'))
optimizer = Adam(lr=0.001)
model.compile(loss='mse', optimizer=optimizer, metrics=["accuracy"])

hist = model.fit(np.array(x_train), np.array(y_train),
    batch_size=64, epochs=200, verbose=1,
    validation_data=(np.array(x_test), np.array(y_test)))

prediction = model.predict(np.array(x_test))
prediction = np.squeeze(prediction)
prediction = np.squeeze(scaler.inverse_transform(prediction.reshape(-1,1)))
prediction = [int(i) for i in prediction]

with open('notes_result', 'wb') as filepath:
    pickle.dump(prediction, filepath)

with open('trainHistoryDict', 'wb') as file_pi:
    pickle.dump(hist.history, file_pi)
