import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib

# Load Dataset
data_gen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
train_gen = data_gen.flow_from_directory('../dataset/flowers/train', color_mode='grayscale')
valid_gen = data_gen.flow_from_directory('../dataset/flowers/validation', color_mode='grayscale')

# Create the model

model = Sequential()
model.add(Conv2D(8,4,4, activation='relu', input_shape=(256,256,1))) # Shape default from flow_from_directory and grayscale
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16,2,2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(8,activation='relu'))
model.add(Dense(5,activation=(tf.nn.softmax)))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=100, epochs=50, validation_steps=len(valid_gen))

# save the plot
matplotlib.use('Agg')
# Accuracy plot
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('../plots/first_cnn_acc.pdf')
plt.close()
# Loss plot
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('../plots/first_cnn_loss.pdf')
