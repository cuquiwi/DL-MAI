from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, Activation
import numpy as np

# Based upon
# https://towardsdatascience.com/generating-pokemon-inspired-music-from-neural-networks-bc240014132
# https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5

n_vocab = 7 # ABCDEFG

class Generator:
	def __init__(self, input="network_input.npy", output="network_output.npy"):
		self.network_input = None
		self.network_output = None
		self.load(input, output)
		self.model = self.build()
		print("Generator initialized")

	def load(self, input="network_input.npy", output="network_output.npy"):
		self.network_input = np.load(input)
		self.network_output = np.load(output)


	def build(self):
		model = Sequential()

		#### Model architecture ###
		model = Sequential()
		model.add(LSTM(
			512,
			input_shape=(self.network_input.shape[1], self.network_input.shape[2]),
			return_sequences=True
		))
		model.add(Dropout(0.3))
		model.add(LSTM(512, return_sequences=True))
		model.add(Dropout(0.3))
		model.add(LSTM(512))
		model.add(Dense(256))
		model.add(Dropout(0.3))
		model.add(Dense(n_vocab))
		model.add(Activation('softmax'))
		model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

		return model

	def train(self):
		self.model.fit(self.network_input, self.network_output, epochs=200, batch_size=64)
		pass

	def validate(self):
		pass

	def test(self):
		pass
