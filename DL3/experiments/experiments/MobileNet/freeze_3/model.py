import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping

name = "MobileNet_3"


### Data

batch_size = 50
num_of_test_samples = 400

# Load Dataset
data_gen = ImageDataGenerator(
        rotation_range=20,
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2)
train_gen = data_gen.flow_from_directory('../../../data/train', color_mode='rgb',
        batch_size=batch_size, target_size=(224,224))
valid_gen = data_gen.flow_from_directory('../../../data/validation', color_mode='rgb',
        batch_size=batch_size, target_size=(224, 224))

early = EarlyStopping(monitor='val_acc', min_delta=0,
                    patience=10, verbose=1, mode='auto')


### Modeling

border = 1
border = -border

base_model = keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
x = base_model.layers[-1].output
predictions = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[:-3]:
    layer.trainable = False

    


model.summary()


### Train the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit_generator(train_gen, validation_data=valid_gen,
        epochs=100, steps_per_epoch=len(train_gen), 
        validation_steps=len(valid_gen), callbacks=[early])



### Saving of all data
# Open the file
with open(name + '.summary','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
with open(name + ".history", 'wb') as file_pi:
    pickle.dump(hist.history, file_pi)
# serialize model to JSON
model_json = model.to_json()
with open(name + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(name + ".h5")

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(valid_gen, num_of_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(valid_gen.classes, y_pred))
print('Classification Report')
target_names = ["tulip", "sunflower", "rose", "dandelion", "daisy"]
print(classification_report(valid_gen.classes, y_pred, target_names=target_names))