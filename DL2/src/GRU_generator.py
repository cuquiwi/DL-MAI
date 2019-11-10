from keras.models import Sequential
from keras.layers.recurrent import GRU
from keras.layers import Dropout


n_epochs = 100
batch_size = 128
n_layers = 5
units = 20

loss_function = 'binary_crossentropy'
optimizer = 'adam'
class_mode = 'binary'

model = Sequential()

num_units = []
for i in range(n_layers-1):
    num_units.append(units)
    model.add(GRU(input_dim=input_dim, output_dim=num_units[0], activation='tanh', return_sequences=True))
    model.add(Dropout(0.25))
for i in range(num_layers-2):
    model.add(GRU(output_dim=num_units[i+1], activation='tanh', return_sequences=True))
    model.add(Dropout(0.25))
model.add(GRU(output_dim=output_dim, activation='softmax', return_sequences=False))
model.add(Dropout(0.25))

model.compile(loss=loss_function, optimizer=optimizer)
