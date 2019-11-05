import mido, pickle, glob
from mido import MidiFile, MidiTrack, Message
from sklearn.preprocessing import MinMaxScaler
import numpy as np

## Train notes
notes = []
for f in glob.glob('../midi_files/train/*.mid'):
    mid = MidiFile(f) 
    print('Parsing %s'% f)
    for msg in mid:
        if not msg.is_meta and msg.channel == 0 and msg.type == 'note_on':
            data = msg.bytes()
            notes.append(data[1])
       
with open('../data/notes_train2', 'wb') as filepath:
    pickle.dump(notes, filepath)

## Test notes
notes = []
for f in glob.glob('../midi_files/test/*.mid'):
    mid = MidiFile(f) 
    print('Parsing %s'% f)
    for msg in mid:
        if not msg.is_meta and msg.channel == 0 and msg.type == 'note_on':
            data = msg.bytes()
            notes.append(data[1])

with open('../data/notes_test2', 'wb') as filepath:
    pickle.dump(notes, filepath)
