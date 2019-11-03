from music21 import converter, instrument, note, chord
import glob
import numpy as np
import keras
import np_utils
import os

n_vocab = 7 # ABCDEFG

class Preprocessor:
    def __init__(self, data_path=os.getcwd()+"/../dataset"):
        self.data_path = data_path
        self.data = []
        self.network_input = None
        self.network_output = None
        self.load()
        self.paired_sequence()


    def load(self):
        for file in os.listdir(self.data_path):
            file = self.data_path + "/" + file
            print("Processing " + file)
            midi = converter.parse(file)
            parts = instrument.partitionByInstrument(midi)
            if parts:  # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else:  # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    self.data.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    self.data.append('.'.join(str(n) for n in element.normalOrder))

    def paired_sequence(self):
        sequence_length = 100
        # get all pitch names
        pitchnames = sorted(set(item for item in self.data))
        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        network_input = []
        network_output = []
        # create input sequences and the corresponding outputs
        for i in range(0, len(self.data) - sequence_length, 1):
            sequence_in = self.data[i:i + sequence_length]
            sequence_out = self.data[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
        n_patterns = len(network_input)
        # reshape the input into a format compatible with LSTM layers
        network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        self.network_input = network_input / float(n_vocab)
        self.network_output = keras.utils.to_categorical(network_output)

    def save_io(self):
        np.save("network_output", self.network_output)
        np.save("network_input", self.network_input)