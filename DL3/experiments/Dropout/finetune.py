import arg_parser
import sys, os
import numpy as np
from keras import applications, optimizers
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from vgg16_places_365 import VGG16_Places365
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def encode_labels(y_train, y_test):
    """
    Encode the labels in a format suitable for sklearn
    """
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    return y_train, y_test

def load_target_task():
    print('Loading dataset')
    # Set target task
    train_data_dir = '../../data/train'
    validation_data_dir = '../../data/validation'

    # Get num classes in target task
    target_classes = len([name for name in os.listdir(train_data_dir)
                          if os.path.isdir(os.path.join(train_data_dir, name))])
    print('Total target classes:', target_classes)
    # Get num instances in training
    nb_train_samples = 0
    for root, dirs, files in os.walk(train_data_dir):
        nb_train_samples += len(files)
    print('Total train instances:', nb_train_samples)
    # Get num instances in test
    nb_validation_samples = 0
    for root, dirs, files in os.walk(validation_data_dir):
        nb_validation_samples += len(files)
    print('Total validation instances:', nb_validation_samples)
    return target_classes, train_data_dir, validation_data_dir, nb_train_samples, nb_validation_samples

def load_target_task_imgs_labels():
    print('Loading dataset')
    # Define data splits
    train_data_dir = '../../data/train'
    validation_data_dir = '../../data/validation'
    # Train set
    train_images = []
    train_labels = []
    # Use a subset of classes to speed up the process. -1 uses all classes.
    num_classes = -1
    for train_dir in os.listdir(train_data_dir):
        train_dir_path = os.path.join(train_data_dir, train_dir)
        for train_img in os.listdir(train_dir_path):
            train_images.append(os.path.join(train_dir_path, train_img))
            train_labels.append(train_dir)
        num_classes -= 1
        if num_classes == 0:
            break
    # Test set
    test_images = []
    test_labels = []
    num_classes = -1
    for test_dir in os.listdir(validation_data_dir):
        test_dir_path = os.path.join(validation_data_dir, test_dir)
        for test_img in os.listdir(test_dir_path):
            test_images.append(os.path.join(test_dir_path, test_img))
            test_labels.append(test_dir)
        num_classes -= 1
        if num_classes == 0:
            break
    print('Total train images:', len(train_images), ' with their corresponding', len(train_labels), 'labels')
    print('Total test images:', len(test_images), ' with their corresponding', len(test_labels), 'labels')

    return train_images, train_labels, test_images, test_labels

def data_generators(train_data_dir, validation_data_dir, img_height, img_width, batch_size):
    # Initiate the train and test generators with data augumentation
    train_datagen = ImageDataGenerator(
            rescale = 1./255)#,
            #horizontal_flip = True,
            #fill_mode = "nearest",
            #zoom_range = 0.3,
            #width_shift_range = 0.3,
            #height_shift_range=0.3,
            #rotation_range=30)

    val_datagen = ImageDataGenerator(
            rescale = 1./255)#,
            #horizontal_flip = True,
            #fill_mode = "nearest",
            #zoom_range = 0.3,
            #width_shift_range = 0.3,
            #height_shift_range=0.3,
            #rotation_range=30)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size = (img_height, img_width),
            batch_size = batch_size,
            class_mode = "categorical")

    validation_generator = val_datagen.flow_from_directory(
            validation_data_dir,
            target_size = (img_height, img_width),
            batch_size = batch_size,
            shuffle = False,
            class_mode = "categorical")
    return train_generator, validation_generator

def plot_learning_curves(history):
    #Accuracy plot
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.title('Training and validation accuracy')
    plt.savefig('./fine_tuning_accuracy.pdf')
    plt.close()
    #Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.title('Training and validation loss')
    plt.savefig('./fine_tuning_loss.pdf')
    return

def std_mc_acc(ground_truth, prediction):
    """Standard average multiclass prediction accuracy
    """
    y_ok = prediction == ground_truth
    acc = []
    for unique_y in np.unique(ground_truth):
        acc.append(np.sum(y_ok[ground_truth == unique_y]) * 1.0 / np.sum(ground_truth == unique_y))
    return np.mean(acc)

if __name__ == "__main__":
    args = vars(arg_parser.parser.parse_args())
    source = args['source_model']


    img_width, img_height = 224, 224
    if source == 'VGG16_ImageNet':
        model = applications.VGG16(weights="imagenet",include_top=False,
                                    input_shape=(img_width, img_height, 3))
    elif source == 'VGG16_Places':
        model = VGG16_Places365(include_top=False, weights='places', 
                                input_shape=(img_width, img_height, 3))
    else:
        sys.stdout.write("Source model specified ", source_model,
                         "not recognized. Try: ", "VGG16_ImageNet", ", VGG16_Places")
        sys.exit(1)
    
    # Load Dataset
    target_classes, train_data_dir, validation_data_dir, nb_train_samples, \
    nb_validation_samples = load_target_task()
    _, train_labels, _, test_labels = load_target_task_imgs_labels()
    train_labels, test_labels = encode_labels(train_labels, test_labels)

    # Set training parameters
    batch_size = 64
    epochs = 30

    # Freeze the layers which you don't want to train. VGG16 has 18 conv and pooling layers 
    for layer in model.layers[:18]:
        layer.trainable = False
    # 0 input_layer.InputLayer
    # 1 convolutional.Conv2D
    # 2 convolutional.Conv2D
    # 3 pooling.MaxPooling2D
    # 4 convolutional.Conv2D
    # 5 convolutional.Conv2D
    # 6 pooling.MaxPooling2D
    # 7 convolutional.Conv2D
    # 8 convolutional.Conv2D
    # 9 convolutional.Conv2D
    # 10 pooling.MaxPooling2D
    # 11 convolutional.Conv2D
    # 12 convolutional.Conv2D
    # 13 convolutional.Conv2D
    # 14 pooling.MaxPooling2D
    # 15 convolutional.Conv2D
    # 16 convolutional.Conv2D
    # 17 convolutional.Conv2D
    # 18 pooling.MaxPooling2D

    # Adding custom layers on top
    x = model.output
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    predictions = Dense(target_classes, activation="softmax")(x)

    # creating the final model 
    model_final = Model(inputs=model.input, output=predictions)

    # compile the model 
    model_final.compile(loss="categorical_crossentropy", 
                        optimizer=optimizers.SGD(lr=0.0005, momentum=0.9),
                        metrics=["accuracy"])
    

    # Initiate the train and test generators with data augumentation 
    train_data_dir = '../../data/train'
    validation_data_dir = '../../data/validation'
    train_generator, validation_generator = data_generators(train_data_dir,
                                                            validation_data_dir, img_height, img_width, batch_size)

    early = EarlyStopping(monitor='val_acc', min_delta=0,
                        patience=10, verbose=1, mode='auto')

    # Train the model 
    history = model_final.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[early])
    
    plot_learning_curves(history)
    
    # Get predicted labels of validation set
    predicted_labels = model_final.predict_generator(validation_generator, steps=nb_validation_samples / batch_size)
    predicted_labels = predicted_labels.argmax(axis=-1)
    le = LabelEncoder()
    le.fit(test_labels)
    predicted_labels = le.inverse_transform(predicted_labels)

    # Print results
    print(accuracy_score(test_labels, predicted_labels))