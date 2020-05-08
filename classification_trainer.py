import os
import cv2
import random
import time

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.callbacks import TensorBoard

NAME = "Transform-vector-gis-classify"

tb = TensorBoard(log_dir='logs/{}'.format(NAME))


def create_training_data(categories, data_dir, img_size, training_data):
    for category in categories:
        print("Image Loading from " + category + "...")
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([resized_array, class_num])
            except Exception as e:
                print("Image Failed to Load")
                pass
    random.shuffle(training_data)


def train_save_model(training_data, img_size):
    print("starting to train data...")

    X = []  # images
    y = []  # labels

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, img_size, img_size, 1)
    y = np.array(y)

    X = tf.keras.utils.normalize(X, axis=1)

    model = Sequential()

    model.add(Flatten(input_shape=X.shape[1:]))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, batch_size=32, epochs=20, validation_split=0.1, callbacks=[tb])
    print(model.summary())
    return model


def train():
    data_dir = "D:\FYP\DataSets\DataCategorised"
    categories = ["LandClassificationMaps", "Satellite", "Scanned"]

    img_size = 250
    training_data = []

    print("starting...")
    create_training_data(categories, data_dir, img_size, training_data)

    print("data loaded!")
    model = train_save_model(training_data, img_size)

    model.save('classification_model/gis_classify.h5')


train()
