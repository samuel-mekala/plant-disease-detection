"""
Plant Disease Detection — GoogleNet (Inception V1)
Based on SDP Report Appendix Code
Best performing model: 99.10% accuracy

Authors: Satyala Murali Karthik, Mekala Samuel, Kurmala Bhanu Prakash
VIT-AP University, December 2024
"""

import os
import numpy as np
import urllib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Flatten, Conv2D, MaxPooling2D,
                                     AveragePooling2D, Concatenate, Dropout,
                                     Input, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score)
from PIL import Image

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

def inceptionnet(x, filters):
    layer1 = Conv2D(filters[0], (1, 1), strides=1, padding="same", activation="relu")(x)

    layer2 = Conv2D(filters[1][0], (1, 1), strides=1, padding="same", activation="relu")(x)
    layer2 = Conv2D(filters[1][1], (3, 3), strides=1, padding="same", activation="relu")(layer2)

    layer3 = Conv2D(filters[2][0], (1, 1), strides=1, padding="same", activation="relu")(x)
    layer3 = Conv2D(filters[2][1], (5, 5), strides=1, padding="same", activation="relu")(layer3)

    layer4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding="same")(x)
    layer4 = Conv2D(filters[3], (1, 1), strides=1, padding="same", activation="relu")(layer4)

    return Concatenate(axis=-1)([layer1, layer2, layer3, layer4])

def helperfunction(x, name=None, num_classes=38):
    layer = AveragePooling2D(pool_size=(5, 5), strides=3, padding="valid")(x)
    layer = Conv2D(128, (1, 1), strides=1, padding="same", activation="relu")(layer)
    layer = Flatten()(layer)
    layer = Dense(255, activation="relu")(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(num_classes, activation="softmax", name=name)(layer)
    return layer

def build_googlenet(input_shape=(120, 120, 3), num_classes=38):
    inputlayer = Input(shape=input_shape)

    layer = Conv2D(64, (7, 7), strides=1, padding="same", activation="relu")(inputlayer)
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(layer)
    layer = BatchNormalization()(layer)

    layer = Conv2D(64, (1, 1), strides=1, padding="same", activation="relu")(layer)
    layer = Conv2D(192, (3, 3), strides=1, padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(layer)

    layer = inceptionnet(layer, [64, (96, 128), (16, 32), 32])
    layer = inceptionnet(layer, [128, (128, 192), (32, 96), 64])
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(layer)

    layer = inceptionnet(layer, [192, (96, 208), (16, 48), 64])
    final_0 = helperfunction(layer, name="final_layer_0", num_classes=num_classes)

    layer = inceptionnet(layer, [160, (112, 224), (24, 64), 64])
    layer = inceptionnet(layer, [128, (128, 256), (24, 64), 64])
    layer = inceptionnet(layer, [112, (144, 288), (32, 64), 64])
    final_1 = helperfunction(layer, name="final_layer_1", num_classes=num_classes)

    layer = inceptionnet(layer, [256, (160, 320), (32, 128), 128])
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(layer)

    layer = inceptionnet(layer, [256, (160, 320), (32, 128), 128])
    layer = inceptionnet(layer, [384, (192, 384), (48, 128), 128])
    layer = AveragePooling2D(pool_size=(7, 7), strides=1, padding="same")(layer)

    layer = Flatten()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(256, activation="linear")(layer)
    final_2 = Dense(num_classes, activation="softmax", name="final_layer_2")(layer)

    return Model(inputs=inputlayer, outputs=[final_2, final_0, final_1], name="GoogleNet")

def train_googlenet():
    image_path, valid_image_path = get_data_paths()
    print(f"Using training path: {image_path}")
    print(f"Using validation path: {valid_image_path}")

    if os.path.exists(image_path):
        class_names = os.listdir(image_path)
        num_classes = len(class_names)
        print(f"Total classes: {num_classes}")

        batch_size = 32
        train_gen = ImageDataGenerator(
            rescale=1./255, zoom_range=0.2, width_shift_range=0.2,
            height_shift_range=0.2, shear_range=0.2,
            horizontal_flip=True, validation_split=0.2
        )
        valid_gen = ImageDataGenerator(rescale=1./255)
        test_gen = ImageDataGenerator(rescale=1./255)

        train_data = train_gen.flow_from_directory(
            image_path, batch_size=batch_size, class_mode="categorical",
            target_size=(120, 120), color_mode="rgb", shuffle=True
        )
        valid_data = valid_gen.flow_from_directory(
            image_path, batch_size=batch_size, class_mode="categorical",
            target_size=(120, 120), color_mode="rgb", shuffle=True
        )
        test_data = test_gen.flow_from_directory(
            valid_image_path, batch_size=batch_size, class_mode="categorical",
            target_size=(120, 120), color_mode="rgb", shuffle=False
        )

        model = build_googlenet(num_classes=num_classes)
        model.summary()

        model.compile(
            loss=['categorical_crossentropy'] * 3,
            loss_weights=[1, 0.3, 0.3],
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

        os.makedirs("fine_tune_checkpoints", exist_ok=True)
        checkpoint_path = "fine_tune_checkpoints/googlenet_best.h5"
        cb = [
            callbacks.EarlyStopping(monitor="val_loss", patience=3),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, verbose=1, min_lr=1e-7),
            callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=True, monitor="val_loss"),
        ]

        history = model.fit(
            train_data,
            steps_per_epoch=max(1, train_data.samples // batch_size),
            epochs=30, verbose=1, callbacks=cb,
            validation_data=valid_data,
            validation_steps=max(1, valid_data.samples // batch_size),
        )
        model.save("googlenet_plant_disease.h5")
        print("GoogleNet model trained and saved successfully to googlenet_plant_disease.h5")
    else:
        print(f"Dataset path '{image_path}' not found.")

if __name__ == "__main__":
    train_googlenet()