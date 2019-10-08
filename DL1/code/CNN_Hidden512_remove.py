import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

batch_size = 50

# Load Dataset
data_gen = ImageDataGenerator(
        rotation_range=20,
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2)
train_gen = data_gen.flow_from_directory('../dataset/flowers/train', color_mode='rgb',
        batch_size=batch_size, target_size=(100,100))
valid_gen = data_gen.flow_from_directory('../dataset/flowers/validation', color_mode='rgb',
        batch_size=batch_size, target_size=(100, 100))

# Create the model

model = Sequential()
model.add(Conv2D(64,3,3, activation='relu', input_shape=(100,100,3)))
model.add(Conv2D(64,3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(128,3,3, activation='relu'))
model.add(Conv2D(128,3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(256,3,3, activation='relu'))
model.add(Conv2D(256,3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(512,3,3, activation='relu'))
model.add(Conv2D(512,3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5,activation=(tf.nn.softmax)))

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit_generator(train_gen, validation_data=valid_gen,
        epochs=100, steps_per_epoch=len(train_gen),
        validation_steps=len(valid_gen))

# save the plot
# Accuracy plot
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('../plots/cnn_Hidden512_remove_acc.pdf')
plt.close()
# Loss plot
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('../plots/cnn_Hidden512_remove_loss.pdf')
