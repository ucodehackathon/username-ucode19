import cv2
import os
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras import Sequential
MAX_NB_CLASSES = 2

def extract_images(video_input_file_path, image_output_dir_path):
    if os.path.exists(image_output_dir_path):
        return
    count = 0
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        if success:
            cv2.imwrite(image_output_dir_path + os.path.sep + "frame%d.jpg" % count, image)  # save frame as JPEG file
            count = count + 1


def extract_features(video_input_file_path, feature_output_file_path):
    if os.path.exists(feature_output_file_path):
        return np.load(feature_output_file_path)
    count = 0
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    features = []
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        if success:
            img = cv2.resize(image, (240, 240), interpolation=cv2.INTER_AREA)
            features.append(image)
            count = count + 1
    unscaled_features = np.array(features)
    print(unscaled_features.shape)
    np.save(feature_output_file_path, unscaled_features)
    return unscaled_features


def extract_videos_for_conv2d(video_input_file_path, feature_output_file_path, max_frames):
    if feature_output_file_path is not None:
        if os.path.exists(feature_output_file_path):
            return np.load(feature_output_file_path)
    count = 0
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    features = []
    success = True
    while success and count < max_frames:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        if success:
            image = cv2.resize(image, (240, 240), interpolation=cv2.INTER_AREA)
            channels = image.shape[2]
            for channel in range(channels):
                features.append(image[:, :, channel])
            count = count + 1
    unscaled_features = np.array(features)
    unscaled_features = np.transpose(unscaled_features, axes=(1, 2, 0))
    print(unscaled_features.shape)
    if feature_output_file_path is not None:
        np.save(feature_output_file_path, unscaled_features)
    return unscaled_features



def scan_and_extract_features(data_dir_path, data_set_name=None):
    input_data_dir_path = data_dir_path + '/' + data_set_name
    output_feature_data_dir_path = data_dir_path + '/' + data_set_name + '-Features'
    
    if not os.path.exists(output_feature_data_dir_path):
        os.makedirs(output_feature_data_dir_path)

    y_samples = []
    x_samples = []

    dir_count = 0
    for f in os.listdir(input_data_dir_path):
        file_path = input_data_dir_path + os.path.sep + f
        if not os.path.isfile(file_path):
            output_dir_name = f
            output_dir_path = output_feature_data_dir_path + os.path.sep + output_dir_name
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + ff
                output_feature_file_path = output_dir_path + os.path.sep + ff.split('.')[0] + '.npy'
                x = extract_features(video_file_path, output_feature_file_path)
                y = f
                y_samples.append(y)
                x_samples.append(x)

        if dir_count == MAX_NB_CLASSES:
            break

    return x_samples, y_samples


def scan_and_extract_videos_for_conv2d(data_dir_path, data_set_name=None, max_frames=None):
    if max_frames is None:
        max_frames = 10

    input_data_dir_path = data_dir_path + '/' + data_set_name
    output_feature_data_dir_path = data_dir_path + '/' + data_set_name + '-Conv2d'

    if not os.path.exists(output_feature_data_dir_path):
        os.makedirs(output_feature_data_dir_path)

    y_samples = []
    x_samples = []

    dir_count = 0
    for f in os.listdir(input_data_dir_path):
        file_path = input_data_dir_path + os.path.sep + f
        if not os.path.isfile(file_path):
            output_dir_name = f
            output_dir_path = output_feature_data_dir_path + os.path.sep + output_dir_name
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + ff
                output_feature_file_path = output_dir_path + os.path.sep + ff.split('.')[0] + '.npy'
                x = extract_videos_for_conv2d(video_file_path, output_feature_file_path, max_frames)
                y = f
                y_samples.append(y)
                x_samples.append(x)

        if dir_count == MAX_NB_CLASSES:
            break

    return x_samples, y_samples

	from matplotlib import pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def most_informative_feature_for_binary_classification(vectorizer, classifier, n=100):
    """
    See: https://stackoverflow.com/a/26980472
    Identify most important features if given a vectorizer and binary classifier. Set n to the number
    of weighted features you would like to show. (Note: current implementation merely prints and does not
    return top classes.)
    """

    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)

    print()

    for coef, feat in reversed(topn_class2):
        print(class_labels[1], coef, feat)


def plot_history_2win(history):
    plt.subplot(211)
    plt.title('Accuracy')
    plt.plot(history.history['acc'], color='g', label='Train')
    plt.plot(history.history['val_acc'], color='b', label='Validation')
    plt.legend(loc='best')

    plt.subplot(212)
    plt.title('Loss')
    plt.plot(history.history['loss'], color='g', label='Train')
    plt.plot(history.history['val_loss'], color='b', label='Validation')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()


def create_history_plot(history, model_name, metrics=None):
    plt.title('Accuracy and Loss (' + model_name + ')')
    if metrics is None:
        metrics = {'acc', 'loss'}
    if 'acc' in metrics:
        plt.plot(history.history['acc'], color='g', label='Train Accuracy')
        plt.plot(history.history['val_acc'], color='b', label='Validation Accuracy')
    if 'loss' in metrics:
        plt.plot(history.history['loss'], color='r', label='Train Loss')
        plt.plot(history.history['val_loss'], color='m', label='Validation Loss')
    plt.legend(loc='best')

    plt.tight_layout()


def plot_history(history, model_name):
    create_history_plot(history, model_name)
    plt.show()


def plot_and_save_history(history, model_name, file_path, metrics=None):
    if metrics is None:
        metrics = {'acc', 'loss'}
    create_history_plot(history, model_name, metrics)
    plt.savefig(file_path)
	
BATCH_SIZE = 32
NUM_EPOCHS = 20

from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import tensorflow as tf

def generate_batch(x_samples, y_samples):
    num_batches = len(x_samples) // BATCH_SIZE

    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            yield np.array(x_samples[start:end]), y_samples[start:end]


class CnnRNNVideoClassifier(object):
    model_name = 'cnn'

    def __init__(self):
        self.img_width = None
        self.img_height = None
        self.img_channels = None
        self.nb_classes = None
        self.labels = None
        self.labels_idx2word = None
        self.model = None
        self.expected_frames = None

    def create_model(self, input_shape, nb_classes):
        model = Sequential()
        model.add(Conv2D(filters=32, input_shape=input_shape, padding='same', kernel_size=(3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=32, padding='same', kernel_size=(3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(rate=0.25))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=64, padding='same', kernel_size=(3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(rate=0.25))

        model.add(Flatten())
        model.add(Dense(units=512))
        model.add(Activation('relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=nb_classes))
        model.add(Activation('sigmoid'))

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + CnnRNNVideoClassifier.model_name + '-weights.h5'

    @staticmethod
    def get_file_path(model_dir_path):
        return model_dir_path + '/' + CnnRNNVideoClassifier.model_name + '-architecture.json'

    def load_model(self, weight_file_path):

        self.img_width = 1080
        self.img_height = 1920
        self.nb_classes = 2
        self.labels = thisdict = {
            1:"Alex",
            2: "Gabriel"
        }
        self.expected_frames = 30
        self.labels_idx2word = dict([(idx, word) for word, idx in self.labels.items()])
        self.model = self.create_model(
            input_shape=(self.img_width, self.img_height, self.expected_frames),
            nb_classes=self.nb_classes)
        self.model.load_weights(weight_file_path)

    def predict(self, video_file_path):
        x = extract_videos_for_conv2d(video_file_path, None, self.expected_frames)
        frames = x.shape[2]
        if frames > self.expected_frames:
            x = x[:, :, 0:self.expected_frames]
        elif frames < self.expected_frames:
            temp = np.zeros(shape=(x.shape[0], x.shape[1], self.expected_frames))
            temp[:, :, 0:frames] = x
            x = temp
        predicted_class = np.argmax(self.model.predict(np.array([x]))[0])
        predicted_label = self.labels_idx2word[predicted_class]
        return predicted_label

    def fit(self, data_dir_path, model_dir_path, epochs=NUM_EPOCHS, data_set_name='UCF-101', max_frames=10,
            test_size=0.5,
            random_state=42):

        weight_file_path = self.get_weight_file_path(model_dir_path)
        file_path = self.get_file_path(model_dir_path)

        self.labels = dict()
        x_samples, y_samples = scan_and_extract_videos_for_conv2d(data_dir_path,
                                                                  max_frames=max_frames,
                                                                  data_set_name=data_set_name)
        self.img_width, self.img_height, _ = x_samples[0].shape
        frames_list = []
        for x in x_samples:
            frames = x.shape[2]
            frames_list.append(frames)
            max_frames = max(frames, max_frames)
        self.expected_frames = int(np.mean(frames_list))
        print('max frames: ', max_frames)
        print('expected frames: ', self.expected_frames)
        for i in range(len(x_samples)):
            x = x_samples[i]
            frames = x.shape[2]
            if frames > self.expected_frames:
                x = x[:, :, 0:self.expected_frames]
                x_samples[i] = x
            elif frames < self.expected_frames:
                temp = np.zeros(shape=(x.shape[0], x.shape[1], self.expected_frames))
                temp[:, :, 0:frames] = x
                x_samples[i] = temp
        for y in y_samples:
            if y not in self.labels:
                self.labels[y] = len(self.labels)
        print(self.labels)
        for i in range(len(y_samples)):
            y_samples[i] = self.labels[y_samples[i]]

        self.nb_classes = len(self.labels)

        y_samples = np_utils.to_categorical(y_samples, self.nb_classes)


        self.model = self.create_model(input_shape=(self.img_width, self.img_height, self.expected_frames),
                                  nb_classes=self.nb_classes)
        #open(file_path, 'w+').write(model.to_json())
        
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_samples, y_samples, test_size=test_size,
                                                        random_state=random_state)
        
        Xtrain=tf.keras.utils.normalize(Xtrain,1,1)
        Xtest=tf.keras.utils.normalize(Xtest,1,1)
        train_gen = generate_batch(Xtrain, Ytrain)
        test_gen = generate_batch(Xtest, Ytest)

        train_num_batches = len(Xtrain)
        test_num_batches = len(Xtest)

        print('start fit_generator')
        checkpoint = ModelCheckpoint(filepath=weight_file_path, save_best_only=True)

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                      epochs=epochs,
                                      verbose=1, validation_data=test_gen, validation_steps=test_num_batches,
                                      callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

        result = self.model.predict(Xtest)
        for i in range(0, len(result)):
          print ('Ejemplo '+str(i))
          print('Real Alex: %.f Estimación Alex: %.4f' % (result[i][0], Ytest[i][0]))
          print('Real Gabriel: %.f Estimación Gabriel: %.4f' % (result[i][1], Ytest[i][1]))
        return history

    def save_graph(self, to_file):
        plot_model(self.model, to_file=to_file)
	
import zipfile
	
!wget --no-check-certificate \
   'URLZIP' -O \
   /tmp/football-dataVid.zip
local_zip = '/tmp/football-dataVid.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()
	
NUM_EPOCHS = 20
BATCH_SIZE = 2
#TRAINING
import numpy as np
from keras import backend as K
import os

K.set_image_dim_ordering('tf')

data_set_name = 'FootballVid'
input_dir_path = '/tmp'
output_dir_path = '/tmp/models/' + data_set_name 
report_dir_path = '/tmp/reports/' + data_set_name 
if not os.path.exists('/tmp/models'):
    os.makedirs('/tmp/models')
if not os.path.exists('/tmp/reports'):
    os.makedirs('/tmp/reports')
if not os.path.exists('/tmp/models/'+data_set_name):
    os.makedirs('/tmp/models/'+data_set_name)
if not os.path.exists('/tmp/reports/'+data_set_name):
    os.makedirs('/tmp/reports/'+data_set_name)
np.random.seed(42)

classifier = CnnRNNVideoClassifier()

history = classifier.fit(data_dir_path=input_dir_path, model_dir_path=output_dir_path, data_set_name=data_set_name)

plot_and_save_history(history, CnnRNNVideoClassifier.model_name,
                      report_dir_path + '/' + CnnRNNVideoClassifier.model_name + '-history.png')