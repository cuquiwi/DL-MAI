import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib

batch_size = 100

# Load Dataset
data_gen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2)
train_gen = data_gen.flow_from_directory('../dataset/flowers/train', color_mode='rgb', batch_size=batch_size)
valid_gen = data_gen.flow_from_directory('../dataset/flowers/validation', color_mode='rgb', batch_size=batch_size)

# Create the model

model = Sequential()
model.add(Conv2D(16,16,16, activation='relu', input_shape=(256,256,3))) # Shape default from flow_from_directory
model.add(Conv2D(32,8,8, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,4,4, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(5,activation=(tf.nn.softmax)))

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit_generator(train_gen, validation_data=valid_gen,
        epochs=50, steps_per_epoch=len(train_gen), 
        validation_steps=len(valid_gen))

# save the plot
matplotlib.use('Agg')
# Accuracy plot
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('../plots/2_cnn_acc.pdf')
plt.close()
# Loss plot
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('../plots/2_cnn_loss.pdf')
