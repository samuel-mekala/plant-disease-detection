"""
Plant Disease Detection — All Models
VGG16, VGG19, ResNet50, LeNet-5, DenseNet

Run any section independently or use train_model() to switch between architectures.

SDP Report: "Plant Diseases Detection Using Deep Learning Techniques"
Authors: Satyala Murali Karthik, Mekala Samuel, Kurmala Bhanu Prakash
VIT-AP University, December 2024
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten,
    Dropout, BatchNormalization, Activation, Input,
    ZeroPadding2D, Add, GlobalAveragePooling2D, Concatenate
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras import callbacks as K_callbacks
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)
from keras.initializers import glorot_uniform

# ─── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR  = "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
TRAIN_DIR = DATA_DIR + "/train"
VALID_DIR = DATA_DIR + "/valid"

BATCH_SIZE = 128

# ─── Shared Data Generators ───────────────────────────────────────────────────

def get_generators(img_size=(224, 224)):
    train_gen = ImageDataGenerator(
        rescale=1./255, shear_range=0.2, zoom_range=0.2,
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        horizontal_flip=True, validation_split=0.2
    )
    val_gen = ImageDataGenerator(rescale=1./255)

    training_set = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=img_size, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training'
    )
    valid_set = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=img_size, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', shuffle=False
    )
    test_set = val_gen.flow_from_directory(
        VALID_DIR, target_size=img_size, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )
    return training_set, valid_set, test_set


def get_callbacks(checkpoint_path):
    return [
        K_callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                  restore_best_weights=True),
        K_callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=2, verbose=1, min_lr=1e-7),
        K_callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True,
                                    save_weights_only=True, monitor='val_loss'),
    ]


def plot_history(hist, name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist['accuracy'],     label='Train')
    plt.plot(hist['val_accuracy'], label='Val')
    plt.title(f'{name} – Accuracy'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist['loss'],     label='Train')
    plt.plot(hist['val_loss'], label='Val')
    plt.title(f'{name} – Loss'); plt.legend()
    plt.savefig(f"{name.lower().replace(' ', '_')}_curves.png", dpi=150)
    plt.show()


def evaluate_model(model, test_set):
    y_pred = np.argmax(model.predict(test_set), axis=1)
    y_true = test_set.classes
    li = list(test_set.class_indices.keys())
    print(classification_report(y_true, y_pred, target_names=li))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    plt.figure(figsize=(30, 30))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d',
                cmap='Blues', xticklabels=li, yticklabels=li)
    plt.title('Confusion Matrix'); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# VGG-16
# ═══════════════════════════════════════════════════════════════════════════════

def build_vgg16():
    base = VGG16(weights='imagenet', include_top=False,
                 input_tensor=Input(shape=(224, 224, 3)))
    for layer in base.layers:
        layer.trainable = False

    x = Flatten()(base.output)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    out = Dense(38, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_vgg16():
    training_set, valid_set, test_set = get_generators()
    model = build_vgg16()
    model.summary()

    history = model.fit(
        training_set,
        steps_per_epoch=training_set.samples // BATCH_SIZE,
        validation_data=valid_set,
        validation_steps=valid_set.samples // BATCH_SIZE,
        epochs=10
    )
    model.save("VGG16Model.h5")
    plot_history(history.history, "VGG-16")
    evaluate_model(model, test_set)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# VGG-19 (custom built)
# ═══════════════════════════════════════════════════════════════════════════════

def build_vgg19():
    model = keras.Sequential([
        Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(224,224,3)),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(), MaxPooling2D((2,2)),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(), MaxPooling2D((2,2)),

        Conv2D(256, (3,3), activation='relu', padding='same'),
        Conv2D(256, (3,3), activation='relu', padding='same'),
        Conv2D(256, (3,3), activation='relu', padding='same'),
        Conv2D(256, (3,3), activation='relu', padding='same'),
        BatchNormalization(), MaxPooling2D((2,2)),

        Conv2D(512, (3,3), activation='relu', padding='same'),
        Conv2D(512, (3,3), activation='relu', padding='same'),
        Conv2D(512, (3,3), activation='relu', padding='same'),
        Conv2D(512, (3,3), activation='relu', padding='same'),
        BatchNormalization(), MaxPooling2D((2,2)),

        Conv2D(512, (3,3), activation='relu', padding='same'),
        Conv2D(512, (3,3), activation='relu', padding='same'),
        Conv2D(512, (3,3), activation='relu', padding='same'),
        Conv2D(512, (3,3), activation='relu', padding='same'),
        BatchNormalization(), MaxPooling2D((2,2)),

        Flatten(),
        Dense(4096, activation='relu'), Dropout(0.3),
        Dense(4096, activation='relu'), Dropout(0.2),
        Dense(38,   activation='softmax'),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_vgg19():
    training_set, valid_set, test_set = get_generators()
    model = build_vgg19()
    model.summary()

    history = model.fit(
        training_set,
        epochs=20,
        steps_per_epoch=len(training_set),
        validation_data=valid_set,
        validation_steps=len(valid_set),
        callbacks=get_callbacks("checkpoints/vgg19_best/")
    )
    model.save("VGG19Model.h5")
    plot_history(history.history, "VGG-19")
    evaluate_model(model, test_set)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# ResNet-50 (transfer learning from Keras applications)
# ═══════════════════════════════════════════════════════════════════════════════

def build_resnet50():
    from tensorflow.keras.applications.resnet50 import preprocess_input

    base = ResNet50(include_top=False, input_shape=(224, 224, 3))
    base.trainable = False

    pt = Input(shape=(224, 224, 3))
    x  = preprocess_input(pt)
    x  = base(x, training=False)
    x  = GlobalAveragePooling2D()(x)
    x  = Dense(128, activation='relu')(x)
    x  = Dense(64,  activation='relu')(x)
    out = Dense(38, activation='softmax')(x)

    model = Model(inputs=pt, outputs=out)
    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_resnet50():
    train_gen = ImageDataGenerator(
        shear_range=0.2, zoom_range=0.2,
        width_shift_range=0.2, height_shift_range=0.2
    )
    val_gen = ImageDataGenerator()

    train = train_gen.flow_from_directory(
        TRAIN_DIR, batch_size=32, target_size=(224,224),
        color_mode='rgb', class_mode='categorical', seed=42
    )
    valid = val_gen.flow_from_directory(
        VALID_DIR, batch_size=32, target_size=(224,224),
        color_mode='rgb', class_mode='categorical'
    )

    model = build_resnet50()
    model.summary()

    es  = K_callbacks.EarlyStopping(monitor='val_accuracy', patience=7, mode='auto', verbose=1)
    lr  = K_callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=5, min_lr=0.001, verbose=1)
    mc  = K_callbacks.ModelCheckpoint('/content', monitor='val_accuracy', save_best_only=True, verbose=1)

    model.fit(train, validation_data=valid, epochs=30,
              steps_per_epoch=200, verbose=1, callbacks=[mc, es, lr])
    model.save("RESNET50_PLANT_DISEASE.h5")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# LeNet-5
# ═══════════════════════════════════════════════════════════════════════════════

def build_lenet5():
    model = Sequential([
        Conv2D(6, (5,5), padding="same", activation="tanh", input_shape=(224,224,3)),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2,2), strides=(2,2)),

        Conv2D(16, (5,5), padding="same", activation="tanh"),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2,2), strides=(2,2)),

        Conv2D(120, (5,5), padding="same", activation="tanh"),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2,2), strides=(2,2)),

        Flatten(),
        Dense(120, activation='tanh', kernel_regularizer=l2(0.001)), Dropout(0.3),
        Dense(84,  activation='tanh', kernel_regularizer=l2(0.001)), Dropout(0.2),
        Dense(38,  activation='softmax'),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model


def train_lenet5():
    training_set, valid_set, test_set = get_generators()
    model = build_lenet5()
    model.summary()

    history = model.fit(
        training_set,
        epochs=30,
        steps_per_epoch=len(training_set),
        validation_data=valid_set,
        validation_steps=len(valid_set),
        callbacks=get_callbacks("checkpoints/lenet5_best/")
    )
    test_loss, test_acc = model.evaluate(test_set)
    print(f"Test Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.4f}")
    plot_history(history.history, "LeNet-5")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# DenseNet (custom)
# ═══════════════════════════════════════════════════════════════════════════════

from keras.layers import concatenate

def conv_layer(x, filters):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3,3), kernel_initializer='he_uniform',
               padding='same', use_bias=False)(x)
    x = Dropout(0.5)(x)
    return x


def dense_block(block_x, filters, growth_rate, layers_in_block):
    for _ in range(layers_in_block):
        layer = conv_layer(block_x, growth_rate)
        block_x = concatenate([block_x, layer], axis=-1)
        filters += growth_rate
    return block_x, filters


def transition_block(x, filters):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), kernel_initializer='he_uniform',
               padding='same', use_bias=False)(x)
    x = AveragePooling2D((2,2), strides=(2,2))(x)
    return x, filters


def build_densenet(growth_rate=12, dense_block_size=3, layers_in_block=4, classes=38):
    filters = growth_rate * 2
    inputs  = Input(shape=(32, 32, 3))
    x = Conv2D(24, (3,3), kernel_initializer='he_uniform',
               padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

    for block in range(dense_block_size - 1):
        x, filters = dense_block(x, filters, growth_rate, layers_in_block)
        x, filters = transition_block(x, filters)

    x, filters = dense_block(x, filters, growth_rate, layers_in_block)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    out = Dense(classes, activation='softmax')(x)

    model = Model(inputs, out)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_densenet():
    training_set, valid_set, test_set = get_generators(img_size=(32, 32))
    model = build_densenet()
    model.summary()

    history = model.fit(
        training_set,
        epochs=20,
        verbose=1,
        callbacks=get_callbacks("checkpoints/densenet_best/"),
        validation_data=valid_set
    )
    model.save("DenseNetModel.hdf5")
    plot_history(history.history, "DenseNet")
    evaluate_model(model, test_set)
    return model


# ─── Results Summary (from paper) ────────────────────────────────────────────

RESULTS = {
    "GoogleNet (Best)": {"Accuracy": 99.10, "F1": 99.00, "AUC": 99.00},
    "VGG-19":           {"Accuracy": "~97", "F1": "—",   "AUC": "—"},
    "ResNet-50":        {"Accuracy": "~96", "F1": "—",   "AUC": "—"},
    "DenseNet":         {"Accuracy": "~95", "F1": "—",   "AUC": "—"},
    "AlexNet":          {"Accuracy": "~92", "F1": "—",   "AUC": "—"},
    "VGG-16":           {"Accuracy": "~91", "F1": "—",   "AUC": "—"},
    "LeNet-5":          {"Accuracy": "~75", "F1": "—",   "AUC": "—"},
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vgg16","vgg19","resnet50","lenet5","densenet"],
                        default="vgg16")
    args = parser.parse_args()

    dispatch = {
        "vgg16":   train_vgg16,
        "vgg19":   train_vgg19,
        "resnet50":train_resnet50,
        "lenet5":  train_lenet5,
        "densenet":train_densenet,
    }
    dispatch[args.model]()
