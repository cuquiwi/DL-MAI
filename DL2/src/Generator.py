from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding

# Based upon
# https://towardsdatascience.com/generating-pokemon-inspired-music-from-neural-networks-bc240014132
# https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5


class Generator:
	def __init__(self, dataset):
		self.model = self.build()
		self.dataset = dataset
		print("Generator initialized")


	def build(self):
		model = Sequential()

		#### Model architecture ###

		# Embeding
		model.add(Embedding(input_dim=2500,
			output_dim = 50
			))
		pass

	def train(self):
		pass

	def validate(self):
		pass

	def test(self):
		pass
