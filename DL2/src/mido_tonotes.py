import pickle, sys
from mido import MidiFile, MidiTrack, Message
import numpy as np


if len(sys.argv)>0:
    data_test_path = sys.argv[1]
else:
    data_test_path = './notes_result'

with open(data_test_path, 'rb') as f:
    notes = pickle.load(f)


mid = MidiFile()
track = MidiTrack()
t = 0
for note in notes:
    # 147 means note_on
    # 67 is velosity
    note = np.asarray([147, note, 67])
    bytes = note.astype(int)
    msg = Message.from_bytes(bytes[0:3])
    t += 1
    msg.time = t
    track.append(msg)
mid.tracks.append(track)
mid.save('music.mid')

