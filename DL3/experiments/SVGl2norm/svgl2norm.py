import arg_parser
import os, sys
from keras import applications
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from l2norm import l2norm
from vgg16_places_365 import VGG16_Places365
import numpy as np
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

def std_mc_acc(ground_truth, prediction):
    """Standard average multiclass prediction accuracy
    """
    y_ok = prediction == ground_truth
    acc = []
    for unique_y in np.unique(ground_truth):
        acc.append(np.sum(y_ok[ground_truth == unique_y]) * 1.0 / np.sum(ground_truth == unique_y))
    return np.mean(acc)

if __name__ == '__main__':
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

    # Define input and target tensors, that is where we want
    #to enter data, and which activations we wish to extract

    #target_tensors = ['block1_conv1/Relu:0','block1_conv2/Relu:0','block2_conv1/Relu:0','block2_conv2/Relu:0','block3_conv1/Relu:0','block3_conv2/Relu:0','block3_conv3/Relu:0','block4_conv1/Relu:0','block4_conv2/Relu:0','block4_conv3/Relu:0','block5_conv1/Relu:0','block5_conv2/Relu:0','block5_conv3/Relu:0','fc1/Relu:0','fc2/Relu:0']
    #target_tensors = ['fc1/Relu:0']
    '''
    target_tensors = ['block1_conv1','block1_conv2',
                      'block2_conv1','block2_conv2',
                      'block3_conv1','block3_conv2','block3_conv3',
                      'block4_conv1','block4_conv2','block4_conv3',
                      'block5_conv1','block5_conv2','block5_conv3',
                      'fc1','fc2']
    '''
    target_tensors = ['block3_conv2', 'block4_conv3', 'block5_conv1', 'block5_conv2', 'block5_conv3']

     # Load Dataset
    target_classes, train_data_dir, validation_data_dir, nb_train_samples, \
    nb_validation_samples = load_target_task()
    train_images, train_labels, test_images, test_labels = load_target_task_imgs_labels()
    train_labels, test_labels = encode_labels(train_labels, test_labels)

    #Parameters for the extraction procedure
    batch_size = 128
    input_reshape = (224, 224)
    # L2-norm on the train set
    l2norm_features = l2norm(model, train_images, batch_size, target_tensors, input_reshape, source)
    print('Done extracting features of training set. Embedding size:',l2norm_features.shape)

    from sklearn import svm
    #Train SVM with the obtained features.
    clf = svm.LinearSVC()
    print('Training SVM...')
    clf.fit(X = l2norm_features, y = train_labels)
    print('Done training SVM on extracted features of training set')

    # L2-norm on the test set
    l2norm_features = l2norm(model, test_images, batch_size, target_tensors, input_reshape, source_model)
    print('Done extracting features of test set')

    #Test SVM with the test set.
    predicted_labels = clf.predict(l2norm_features)
    print('Done testing SVM on extracted features of test set')

    #Print results
    print(std_mc_acc(test_labels, predicted_labels))