"""
Plant Disease Detection — AlexNet
Based on SDP Report Appendix Code

Authors: Satyala Murali Karthik, Mekala Samuel, Kurmala Bhanu Prakash
VIT-AP University, December 2024
Dataset: New Plant Diseases Dataset (Kaggle) — 87,000+ images, 38 classes
"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPool2D, Activation, Flatten,
                                     Dense, Dropout, BatchNormalization)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def get_data_paths():
    candidates = [
        "data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)",
        "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)",
        "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)"
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train")):
            return os.path.join(c, "train"), os.path.join(c, "valid")
    return candidates[0] + "/train", candidates[0] + "/valid"

def build_alexnet(input_shape=(224, 224, 3), num_classes=38):
    model = Sequential(name="AlexNet")
    model.add(Conv2D(96, (11, 11), strides=(4, 4), padding='valid',
                     kernel_regularizer=l2(0.0005), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same',
                     kernel_regularizer=l2(0.0005)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())

    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                     kernel_regularizer=l2(0.0005)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                     kernel_regularizer=l2(0.0005)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                     kernel_regularizer=l2(0.0005)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

if __name__ == "__main__":
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        strategy = tf.distribute.TPUStrategy(tpu)
    except ValueError:
        strategy = tf.distribute.MirroredStrategy()
    print("Number of accelerators: ", strategy.num_replicas_in_sync)

    train_dir, test_dir = get_data_paths()
    print(f"Using train path: {train_dir}")
    print(f"Using test path: {test_dir}")

    if os.path.exists(train_dir):
        diseases = os.listdir(train_dir)
        num_classes = len(diseases)
        print("Total disease classes are: {}".format(num_classes))

        train_datagen = ImageDataGenerator(
            rescale=1./255, shear_range=0.2, zoom_range=0.2,
            fill_mode="nearest", rotation_range=20,
            width_shift_range=0.2, height_shift_range=0.2,
            horizontal_flip=True, validation_split=0.2
        )
        test_datagen = ImageDataGenerator(
            rescale=1./255, shear_range=0.2, zoom_range=0.2,
            rotation_range=20, horizontal_flip=True
        )

        training_set = train_datagen.flow_from_directory(
            train_dir, target_size=(224, 224), batch_size=128,
            class_mode='categorical', subset='training'
        )
        validation_set = train_datagen.flow_from_directory(
            train_dir, target_size=(224, 224), batch_size=128,
            class_mode='categorical', subset='validation', shuffle=False
        )
        test_set = test_datagen.flow_from_directory(
            test_dir, target_size=(224, 224), batch_size=128, class_mode='categorical'
        )

        label_map = training_set.class_indices
        li = list(label_map.keys())

        with strategy.scope():
            model = build_alexnet(num_classes=num_classes)
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', TopKCategoricalAccuracy(k=1, name="top1")]
            )

        print(model.summary())
        os.makedirs("fine_tune_checkpoints", exist_ok=True)
        checkpoint_path = "fine_tune_checkpoints/alexnet_weights.h5"
        cb = [
            callbacks.EarlyStopping(monitor="val_loss", patience=3),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, verbose=1, min_lr=1e-7),
            callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=True, monitor="val_loss"),
        ]

        history = model.fit(training_set, epochs=20, verbose=1, callbacks=cb, validation_data=validation_set)
        model.load_weights(checkpoint_path)
        model.evaluate(test_set)
        model.save("AlexNetModel.hdf5")
        print("AlexNet model trained and saved successfully!")
    else:
        print(f"Dataset path '{train_dir}' not found.")