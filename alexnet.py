"""
Plant Disease Detection — AlexNet Model
Dataset: New Plant Diseases Dataset (Kaggle, 87,000+ images, 38 classes)

SDP Report: "Plant Diseases Detection Using Deep Learning Techniques"
Authors: Satyala Murali Karthik (21BCB7125), Mekala Samuel (21BCB7145),
         Kurmala Bhanu Prakash (21BCE7701)
VIT-AP University, December 2024
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Model, callbacks
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Activation, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)

# ─── Distributed Strategy (TPU / GPU / CPU) ───────────────────────────────────

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.MirroredStrategy()

print("Number of accelerators:", strategy.num_replicas_in_sync)

# ─── Dataset Paths ────────────────────────────────────────────────────────────
# Download from Kaggle: new-plant-diseases-dataset
# https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

DATA_DIR  = "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
TRAIN_DIR = DATA_DIR + "/train"
TEST_DIR  = DATA_DIR + "/valid"

diseases = os.listdir(TRAIN_DIR)
print("Total disease classes:", len(diseases))

# ─── Data Generators ──────────────────────────────────────────────────────────

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode="nearest",
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    horizontal_flip=True
)

BATCH_SIZE = 128

training_set = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(224, 224),
    batch_size=BATCH_SIZE, class_mode='categorical', subset='training'
)

validation_set = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(224, 224),
    batch_size=BATCH_SIZE, class_mode='categorical',
    subset='validation', shuffle=False
)

test_set = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(224, 224),
    batch_size=BATCH_SIZE, class_mode='categorical'
)

label_map = training_set.class_indices
li = list(label_map.keys())
print("Class mapping:", label_map)

# ─── AlexNet Architecture ─────────────────────────────────────────────────────

with strategy.scope():
    model = Sequential(name="AlexNet")

    # Layer 1: Conv + Pool + BatchNorm
    model.add(Conv2D(96, (11,11), strides=(4,4), padding='valid',
                     kernel_regularizer=l2(0.0005), input_shape=(224,224,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())

    # Layer 2: Conv + Pool + BatchNorm
    model.add(Conv2D(256, (5,5), strides=(1,1), padding='same',
                     kernel_regularizer=l2(0.0005)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())

    # Layer 3
    model.add(Conv2D(384, (3,3), strides=(1,1), padding='same',
                     kernel_regularizer=l2(0.0005)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # Layer 4
    model.add(Conv2D(384, (3,3), strides=(1,1), padding='same',
                     kernel_regularizer=l2(0.0005)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # Layer 5: Conv + Pool + BatchNorm
    model.add(Conv2D(256, (3,3), strides=(1,1), padding='same',
                     kernel_regularizer=l2(0.0005)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

    # FC Layers
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(38, activation='softmax'))   # 38 disease classes

    print(model.summary())

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', TopKCategoricalAccuracy(k=1, name="top1")]
    )

# ─── Callbacks ────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = "checkpoints/alexnet_best/"

early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=3)
reduce_lr      = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                             patience=2, verbose=1, min_lr=1e-7)
model_ckpt     = callbacks.ModelCheckpoint(CHECKPOINT_PATH,
                                           save_weights_only=True,
                                           save_best_only=True,
                                           monitor="val_loss")

# ─── Training ─────────────────────────────────────────────────────────────────

history = model.fit(
    training_set,
    epochs=20,
    verbose=1,
    callbacks=[early_stopping, model_ckpt, reduce_lr],
    validation_data=validation_set
)

model.load_weights(CHECKPOINT_PATH)
model.evaluate(test_set)
model.save("AlexNetModel.hdf5")

# ─── Plots ────────────────────────────────────────────────────────────────────

hist = history.history

def show_plt(metric):
    plt.figure()
    plt.plot(hist[metric],         label=metric)
    plt.plot(hist["val_" + metric],label="val_" + metric)
    plt.ylabel(metric.capitalize())
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(f"alexnet_{metric}.png", dpi=150)
    plt.show()

show_plt("accuracy")
show_plt("loss")

# ─── Evaluation & Confusion Matrix ────────────────────────────────────────────

validation_set.reset()
pred         = model.predict(validation_set)
final_predict = np.argmax(pred, axis=1)
true_data     = validation_set.classes

plt.figure(figsize=(40, 40))
confusion = confusion_matrix(true_data, final_predict)
sns.heatmap(confusion, annot=True, fmt='d', cmap='jet',
            xticklabels=li, yticklabels=li, lw=6)
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('True',      fontsize=20)
plt.title('Confusion Matrix\n', fontsize=20)
plt.savefig("alexnet_confusion_matrix.png", dpi=100)
plt.show()

print(classification_report(true_data, final_predict, target_names=li))
print(f"Accuracy Score: {accuracy_score(true_data, final_predict):.4f}")

# ─── Single Image Prediction ──────────────────────────────────────────────────

from tensorflow.keras.preprocessing import image as keras_image

def predict_single(image_path):
    img = keras_image.load_img(image_path, target_size=(224, 224))
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    pred   = model.predict(arr).flatten()
    idx    = pred.argmax()
    name   = li[idx]
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{name}\n({pred[idx]*100:.1f}%)")
    plt.show()
    return name, pred[idx]

# Example:
# predict_single("/path/to/leaf_image.JPG")
