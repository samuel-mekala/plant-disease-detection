"""
Plant Disease Detection — GoogleNet (Inception V1) Model
Best performing model: 99.10% accuracy

SDP Report: "Plant Diseases Detection Using Deep Learning Techniques"
Authors: Satyala Murali Karthik (21BCB7125), Mekala Samuel (21BCB7145),
         Kurmala Bhanu Prakash (21BCE7701)
VIT-AP University, December 2024
"""

import os
import numpy as np
import urllib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D,
    Concatenate, Dropout, Input, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, precision_recall_curve, auc
)
from sklearn.preprocessing import label_binarize
from PIL import Image

# ─── Dataset Paths ────────────────────────────────────────────────────────────

IMAGE_PATH       = "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/"
TRAIN_IMAGE_PATH = IMAGE_PATH
VALID_IMAGE_PATH = "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/"
TEST_IMAGE_PATH  = "../input/new-plant-diseases-dataset/test/"

class_names = os.listdir(IMAGE_PATH)
print(f"Total classes: {len(class_names)}")

# ─── Data Generators ──────────────────────────────────────────────────────────

BATCH_SIZE = 32

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, zoom_range=0.2, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2,
    horizontal_flip=True, validation_split=0.2
)
valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen  = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_IMAGE_PATH, batch_size=BATCH_SIZE, class_mode="categorical",
    target_size=(120, 120), color_mode="rgb", shuffle=True
)
valid_data = valid_gen.flow_from_directory(
    TRAIN_IMAGE_PATH, batch_size=BATCH_SIZE, class_mode="categorical",
    target_size=(120, 120), color_mode="rgb", shuffle=True
)
test_data = test_gen.flow_from_directory(
    VALID_IMAGE_PATH, batch_size=BATCH_SIZE, class_mode="categorical",
    target_size=(120, 120), color_mode="rgb", shuffle=False
)

train_number = train_data.samples
valid_number = valid_data.samples

# ─── Inception Module ─────────────────────────────────────────────────────────

def inception_module(x, filters):
    # 1×1 branch
    b1 = Conv2D(filters[0], (1,1), strides=1, padding="same", activation="relu")(x)

    # 3×3 branch (1×1 reduce → 3×3)
    b2 = Conv2D(filters[1][0], (1,1), strides=1, padding="same", activation="relu")(x)
    b2 = Conv2D(filters[1][1], (3,3), strides=1, padding="same", activation="relu")(b2)

    # 5×5 branch
    b3 = Conv2D(filters[2][0], (5,5), strides=1, padding="same", activation="relu")(x)
    b3 = Conv2D(filters[2][1], (5,5), strides=1, padding="same", activation="relu")(b3)

    # Pool branch
    b4 = MaxPooling2D(pool_size=(3,3), strides=1, padding="same")(x)
    b4 = Conv2D(filters[3], (1,1), strides=1, padding="same", activation="relu")(b4)

    return Concatenate(axis=-1)([b1, b2, b3, b4])


def auxiliary_classifier(x, name=None):
    x = AveragePooling2D(pool_size=(5,5), strides=3, padding="valid")(x)
    x = Conv2D(128, (1,1), strides=1, padding="same", activation="relu")(x)
    x = Flatten()(x)
    x = Dense(255, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(38, activation="softmax", name=name)(x)
    return x

# ─── GoogleNet Architecture ───────────────────────────────────────────────────

def build_googlenet():
    inputs = Input(shape=(120, 120, 3))

    # Stem
    x = Conv2D(64, (7,7), strides=1, padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (1,1), strides=1, padding="same", activation="relu")(x)
    x = Conv2D(192, (3,3), strides=1, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(x)

    # Inception 3a, 3b
    x = inception_module(x, [64, (96,128),  (16,32),  32])
    x = inception_module(x, [128,(128,192), (32,96),  64])
    x = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(x)

    # Inception 4a, aux0
    x  = inception_module(x, [192,(96,208),  (16,48),  64])
    ax0 = auxiliary_classifier(x, name="aux_output_0")

    # Inception 4b, 4c, 4d, aux1
    x  = inception_module(x, [160,(112,224), (24,64),  64])
    x  = inception_module(x, [128,(128,256), (24,64),  64])
    x  = inception_module(x, [112,(144,288), (32,64),  64])
    ax1 = auxiliary_classifier(x, name="aux_output_1")

    # Inception 4e, 5a, 5b
    x = inception_module(x, [256,(160,320), (32,128), 128])
    x = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(x)
    x = inception_module(x, [256,(160,320), (32,128), 128])
    x = inception_module(x, [384,(192,384), (48,128), 128])
    x = AveragePooling2D(pool_size=(7,7), strides=1, padding="same")(x)

    # Final classifier
    x   = Flatten()(x)
    x   = Dropout(0.5)(x)
    x   = Dense(256, activation="linear")(x)
    out = Dense(38, activation="softmax", name="main_output")(x)

    return Model(inputs=inputs, outputs=[out, ax0, ax1])


model = build_googlenet()
model.summary()

# ─── Compile ──────────────────────────────────────────────────────────────────

optimizer = Adam(learning_rate=0.001)
model.compile(
    loss=["categorical_crossentropy"] * 3,
    loss_weights=[1.0, 0.3, 0.3],
    optimizer=optimizer,
    metrics=["accuracy"]
)

# ─── Callbacks ────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = "checkpoints/googlenet_best/"

early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=3)
reduce_lr      = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                             patience=2, verbose=1, min_lr=1e-7)
model_ckpt     = callbacks.ModelCheckpoint(CHECKPOINT_PATH,
                                           save_weights_only=True,
                                           save_best_only=True,
                                           monitor="val_loss")

# ─── Training ─────────────────────────────────────────────────────────────────

history = model.fit(
    train_data,
    steps_per_epoch=train_number // BATCH_SIZE,
    epochs=30,
    verbose=1,
    callbacks=[early_stopping, model_ckpt, reduce_lr],
    validation_data=valid_data,
    validation_steps=valid_number // BATCH_SIZE
)

model.save("googlenet_plant_disease.h5")

# ─── Plots ────────────────────────────────────────────────────────────────────

hist = history.history

plt.figure()
plt.plot(hist["main_output_accuracy"],     label="Train Accuracy")
plt.plot(hist["val_main_output_accuracy"], label="Val Accuracy")
plt.ylabel("Accuracy"); plt.xlabel("Epochs"); plt.legend()
plt.savefig("googlenet_accuracy.png", dpi=150); plt.show()

plt.figure()
plt.plot(hist["loss"],     label="Train Loss")
plt.plot(hist["val_loss"], label="Val Loss")
plt.ylabel("Loss"); plt.xlabel("Epochs"); plt.legend()
plt.savefig("googlenet_loss.png", dpi=150); plt.show()

# ─── Evaluation ───────────────────────────────────────────────────────────────

pred_main, _, _ = model.predict(test_data)
final_predict   = np.argmax(pred_main, axis=1)
true_data       = test_data.classes

print(f"Accuracy Score: {accuracy_score(true_data, final_predict):.4f}")
print(classification_report(true_data, final_predict, target_names=class_names))

plt.figure(figsize=(40, 40))
confusion = confusion_matrix(true_data, final_predict)
sns.heatmap(confusion, annot=True, fmt='d', cmap='jet',
            xticklabels=class_names, yticklabels=class_names, lw=6)
plt.xlabel('Predicted', fontsize=20); plt.ylabel('True', fontsize=20)
plt.title('Confusion Matrix\n', fontsize=20)
plt.savefig("googlenet_confusion_matrix.png", dpi=100); plt.show()

# ─── Single Image Prediction ──────────────────────────────────────────────────

def predict_single(image_path):
    img  = Image.open(image_path).resize((120, 120))
    arr  = np.array(img) / 255.0
    arr  = np.expand_dims(arr, axis=0)
    probs = model.predict(arr)[0]      # main output
    idx  = np.argmax(probs)
    name = class_names[idx]
    conf = np.max(probs)
    print(f"Predicted: {name}  ({conf*100:.1f}%)")
    plt.figure(figsize=(4, 4)); plt.imshow(img)
    plt.axis('off')
    plt.text(5, 15, f"{name}\n{conf*100:.1f}%", fontsize=10, color='red',
             bbox=dict(facecolor='white', alpha=0.8))
    plt.show()
    return name, conf

# Example:
# predict_single("/path/to/leaf_image.JPG")
