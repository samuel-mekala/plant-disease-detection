"""
Plant Disease Detection — Multiple Models
LeNet-5 | VGG-16 | VGG-19 | ResNet-50 | DenseNet
Based on SDP Report Appendix Code

Authors: Satyala Murali Karthik, Mekala Samuel, Kurmala Bhanu Prakash
VIT-AP University, December 2024

Usage:
  python models.py --model lenet5
  python models.py --model vgg16
  python models.py --model vgg19
  python models.py --model resnet50
  python models.py --model densenet
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPool2D, AveragePooling2D, Dense, Flatten, Dropout,
    BatchNormalization, Activation, Add, Input, GlobalAveragePooling2D,
    ZeroPadding2D, concatenate
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

DATA_DIR = "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")

def get_generators(img_size=(224, 224), batch_size=128):
    train_gen = ImageDataGenerator(
        rescale=1./255, shear_range=0.2, zoom_range=0.2,
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        horizontal_flip=True, validation_split=0.2
    )
    val_gen = ImageDataGenerator(rescale=1./255)

    training_set = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', subset='training'
    )
    valid_set = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', subset='validation', shuffle=False
    )
    test_set = val_gen.flow_from_directory(
        VALID_DIR, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )
    return training_set, valid_set, test_set

def get_callbacks(checkpoint_path):
    return [
        callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, min_lr=1e-7),
        callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=True, monitor='val_loss'),
    ]

def show_plt(hist, metric):
    if metric in hist:
        plt.plot(hist[metric], label=metric)
    if "val_" + metric in hist:
        plt.plot(hist["val_" + metric], label="val_" + metric)
    plt.ylabel(metric.capitalize())
    plt.xlabel("Epochs #")
    plt.legend()
    plt.show()

def evaluate(model, test_set):
    li = list(test_set.class_indices.keys())
    y_pred = np.argmax(model.predict(test_set), axis=1)
    y_true = test_set.classes
    print(f'Accuracy Score: {accuracy_score(y_true, y_pred):.4f}')
    print(classification_report(y_true, y_pred, target_names=li))
    plt.figure(figsize=(30, 30))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d',
                cmap='Blues', xticklabels=li, yticklabels=li)
    plt.title('Confusion Matrix')
    plt.show()

# --- LeNet-5 ---
def build_lenet5(input_shape=(224, 224, 3), num_classes=38):
    model = Sequential([
        Conv2D(6, (5, 5), padding="same", activation="tanh", input_shape=input_shape),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(16, (5, 5), padding="same", activation="tanh"),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(120, (5, 5), padding="same", activation="tanh"),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),

        Flatten(),
        Dense(120, activation='tanh', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(84, activation='tanh', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(num_classes, activation='softmax'),
    ], name="LeNet-5")
    return model

def train_lenet5():
    if not os.path.exists(TRAIN_DIR):
        print(f"Dataset path '{TRAIN_DIR}' not found. Building model architecture for validation.")
        m = build_lenet5()
        print(m.summary())
        return
    training_set, valid_set, test_set = get_generators(img_size=(224, 224))
    model = build_lenet5()
    model.compile(optimizer=Adam(learning_rate=0.001), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    history = model.fit(
        training_set, epochs=30,
        steps_per_epoch=len(training_set),
        validation_data=valid_set,
        validation_steps=len(valid_set),
        callbacks=get_callbacks("checkpoints/lenet5/")
    )
    test_loss, test_acc = model.evaluate(test_set)
    print(f"Test Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.4f}")
    show_plt(history.history, "accuracy")
    show_plt(history.history, "loss")
    evaluate(model, test_set)

# --- VGG-16 ---
def build_vgg16(input_shape=(224, 224, 3), num_classes=38):
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
    for layer in base_model.layers:
        layer.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output, name="VGG16")
    return model

def train_vgg16():
    if not os.path.exists(TRAIN_DIR):
        print(f"Dataset path '{TRAIN_DIR}' not found. Building model architecture for validation.")
        m = build_vgg16()
        print(m.summary())
        return
    training_set, valid_set, test_set = get_generators()
    model = build_vgg16()
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(
        training_set,
        steps_per_epoch=training_set.samples // 128,
        validation_data=valid_set,
        validation_steps=valid_set.samples // 128,
        epochs=10
    )
    model.save("VGG16Model.h5")
    show_plt(history.history, "accuracy")
    show_plt(history.history, "loss")
    evaluate(model, test_set)

# --- VGG-19 ---
def build_vgg19(input_shape=(224, 224, 3), num_classes=38):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(), MaxPool2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(), MaxPool2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(), MaxPool2D((2, 2)),

        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(), MaxPool2D((2, 2)),

        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(), MaxPool2D((2, 2)),

        Flatten(),
        Dense(4096, activation='relu'), Dropout(0.3),
        Dense(4096, activation='relu'), Dropout(0.2),
        Dense(num_classes, activation='softmax'),
    ], name="VGG19")
    return model

def train_vgg19():
    if not os.path.exists(TRAIN_DIR):
        print(f"Dataset path '{TRAIN_DIR}' not found. Building model architecture for validation.")
        m = build_vgg19()
        print(m.summary())
        return
    training_set, valid_set, test_set = get_generators()
    model = build_vgg19()
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(
        training_set, epochs=20,
        steps_per_epoch=len(training_set),
        validation_data=valid_set,
        validation_steps=len(valid_set),
        callbacks=get_callbacks("checkpoints/vgg19/")
    )
    test_loss, test_acc = model.evaluate(test_set)
    print(f"Test Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.4f}")
    model.save("VGG19Model.h5")
    show_plt(history.history, "accuracy")
    show_plt(history.history, "loss")
    evaluate(model, test_set)

# --- ResNet-50 ---
def build_resnet50(input_shape=(224, 224, 3), num_classes=38):
    base_model = ResNet50(include_top=False, input_shape=input_shape)
    base_model.trainable = False
    pt = Input(shape=input_shape)
    x = preprocess_input(pt)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=pt, outputs=out, name="ResNet50")
    return model

def train_resnet50():
    if not os.path.exists(TRAIN_DIR):
        print(f"Dataset path '{TRAIN_DIR}' not found. Building model architecture for validation.")
        m = build_resnet50()
        print(m.summary())
        return
    train_gen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2)
    val_gen = ImageDataGenerator()
    train = train_gen.flow_from_directory(TRAIN_DIR, batch_size=32, target_size=(224, 224), class_mode='categorical', seed=42)
    valid = val_gen.flow_from_directory(VALID_DIR, batch_size=32, target_size=(224, 224), class_mode='categorical')

    model = build_resnet50()
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    cb = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=7, verbose=1, mode='auto'),
        callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=5, min_lr=0.001, verbose=1),
        callbacks.ModelCheckpoint('checkpoints/resnet50/', monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    history = model.fit(train, validation_data=valid, epochs=30, steps_per_epoch=200, verbose=1, callbacks=cb)
    model.save("RESNET50_PLANT_DISEASE.h5")
    show_plt(history.history, "accuracy")
    show_plt(history.history, "loss")

# --- DenseNet ---
def conv_layer(conv_x, filters):
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    conv_x = Conv2D(filters, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(conv_x)
    conv_x = Dropout(0.5)(conv_x)
    return conv_x

def dense_block(block_x, filters, growth_rate, layers_in_block):
    for _ in range(layers_in_block):
        each_layer = conv_layer(block_x, growth_rate)
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate
    return block_x, filters

def transition_block(trans_x, tran_filters):
    trans_x = BatchNormalization()(trans_x)
    trans_x = Activation('relu')(trans_x)
    trans_x = Conv2D(tran_filters, (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=False)(trans_x)
    trans_x = AveragePooling2D((2, 2), strides=(2, 2))(trans_x)
    return trans_x, tran_filters

def dense_net(filters, growth_rate, classes, dense_block_size, layers_in_block, input_shape=(32, 32, 3)):
    input_img = Input(shape=input_shape)
    x = Conv2D(24, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(input_img)
    dense_x = BatchNormalization()(x)
    dense_x = Activation('relu')(dense_x)
    dense_x = MaxPool2D((3, 3), strides=(2, 2), padding='same')(dense_x)
    for _ in range(dense_block_size - 1):
        dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
        dense_x, filters = transition_block(dense_x, filters)
    dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
    dense_x = BatchNormalization()(dense_x)
    dense_x = Activation('relu')(dense_x)
    dense_x = GlobalAveragePooling2D()(dense_x)
    output = Dense(classes, activation='softmax')(dense_x)
    return Model(inputs=input_img, outputs=output, name="DenseNet")

def train_densenet():
    if not os.path.exists(TRAIN_DIR):
        print(f"Dataset path '{TRAIN_DIR}' not found. Building model architecture for validation.")
        m = dense_net(24, 12, 38, 3, 4)
        print(m.summary())
        return
    training_set, valid_set, test_set = get_generators(img_size=(32, 32))
    model = dense_net(24, 12, 38, 3, 4)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        training_set, epochs=20, verbose=1,
        callbacks=get_callbacks("checkpoints/densenet/"),
        validation_data=valid_set,
    )
    model.save("DenseNetModel.hdf5")
    show_plt(history.history, "accuracy")
    show_plt(history.history, "loss")
    evaluate(model, test_set)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lenet5", "vgg16", "vgg19", "resnet50", "densenet"], default="vgg16")
    args = parser.parse_args()

    dispatch = {
        "lenet5": train_lenet5,
        "vgg16": train_vgg16,
        "vgg19": train_vgg19,
        "resnet50": train_resnet50,
        "densenet": train_densenet,
    }
    dispatch[args.model]()