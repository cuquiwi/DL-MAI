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
from keras.applications import *
from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam
import pickle


### Data

batch_size = 50

# Load Dataset
data_gen = ImageDataGenerator(
        rotation_range=20,
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2)
train_gen = data_gen.flow_from_directory('../../data/train', color_mode='rgb',
        batch_size=batch_size, target_size=(224,224))
valid_gen = data_gen.flow_from_directory('../../data/validation', color_mode='rgb',
        batch_size=batch_size, target_size=(224, 224))




### Modeling

base_model = keras.applications.mobilenet.InceptionResNetV2(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
x = base_model.layers[-6].output
predictions = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[:-5]:
    layer.trainable = False


model.summary()

### Train the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit_generator(train_gen, validation_data=valid_gen,
        epochs=100, steps_per_epoch=len(train_gen), 
        validation_steps=len(valid_gen))

with open('trainHistoryDict', 'wb') as file_pi:
    pickle.dump(hist.history, file_pi)