import keras
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import seaborn as sn
import pandas as pd



batch_size = 50
n_batches = 50


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("weights_model.hdf5") 


data_gen = ImageDataGenerator()

gen = data_gen.flow_from_directory('./dataset/validation', color_mode='rgb',batch_size=batch_size, target_size=(100, 100))
#y_pred = model.predict_generator(gen, steps=24)
#y_pred = np.argmax(y_pred, axis=1)
#print(confusion_matrix(gen.classes, y_pred))
Y_pred = model.predict_generator(gen, 400 // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
print('Confusion Matrix')
t = confusion_matrix(gen.classes, y_pred)
print(t)




df_cm = pd.DataFrame(t, index=["sunflower", "rose", "tulip", "dandelion", "daisy"],
                  columns=["sunflower", "rose", "tulip", "dandelion", "daisy"])
sn.heatmap(df_cm, annot=True)