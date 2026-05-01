“””
Plant Disease Detection — GoogleNet (Inception V1)
Based on SDP Report Appendix Code
Best performing model: 99.10% accuracy

Authors: Satyala Murali Karthik, Mekala Samuel, Kurmala Bhanu Prakash
VIT-AP University, December 2024
“””

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
from sklearn.metrics import (classification_report, confusion_matrix,
accuracy_score)
from PIL import Image

# ─── Dataset Paths ───────────────────────────────────────────────────────────

image_path      = “../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/”
train_image_path = image_path
valid_image_path = “../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/”

class_names = os.listdir(image_path)
print(f”Total classes: {len(class_names)}”)

# ─── Data Generators ─────────────────────────────────────────────────────────

batch_size = 32

train_gen = ImageDataGenerator(
rescale=1./255, zoom_range=0.2, width_shift_range=0.2,
height_shift_range=0.2, shear_range=0.2,
horizontal_flip=True, validation_split=0.2
)
valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen  = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
train_image_path, batch_size=batch_size, class_mode=“categorical”,
target_size=(120, 120), color_mode=“rgb”, shuffle=True
)
valid_data = valid_gen.flow_from_directory(
train_image_path, batch_size=batch_size, class_mode=“categorical”,
target_size=(120, 120), color_mode=“rgb”, shuffle=True
)
test_data = test_gen.flow_from_directory(
valid_image_path, batch_size=batch_size, class_mode=“categorical”,
target_size=(120, 120), color_mode=“rgb”, shuffle=False
)

train_number = train_data.samples
valid_number = valid_data.samples

# ─── Inception Module ────────────────────────────────────────────────────────

def inceptionnet(x, filters):
layer1 = Conv2D(filters[0], (1,1), strides=1, padding=“same”, activation=“relu”)(x)

```
layer2 = Conv2D(filters[1][0], (1,1), strides=1, padding="same", activation="relu")(x)
layer2 = Conv2D(filters[1][1], (1,1), strides=1, padding="same", activation="relu")(layer2)

layer3 = Conv2D(filters[2][0], (5,5), strides=1, padding="same", activation="relu")(x)
layers = Conv2D(filters[2][1], (5,5), strides=1, padding="same", activation="relu")(layer3)

layer4 = MaxPooling2D(pool_size=(3,3), strides=1, padding="same")(x)
layer4 = Conv2D(filters[3], (1,1), strides=1, padding="same", activation="relu")(layer4)

return Concatenate(axis=-1)([layer1, layer2, layer3, layer4])
```

def helperfunction(x, name=None):
layer = AveragePooling2D(pool_size=(5,5), strides=3, padding=“valid”)(x)
layer = Conv2D(128, (1,1), strides=1, padding=“same”, activation=“relu”)(layer)
layer = Flatten()(layer)
layer = Dense(255, activation=“relu”)(layer)
layer = Dropout(0.5)(layer)
layer = Dense(38, activation=“softmax”, name=name)(layer)
return layer

# ─── GoogleNet Architecture ──────────────────────────────────────────────────

def googlenet():
inputlayer = Input(shape=(120, 120, 3))

```
layer = Conv2D(64, (7,7), strides=1, padding="same", activation="relu")(inputlayer)
layer = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(layer)
layer = BatchNormalization()(layer)

layer = Conv2D(64,  (1,1), strides=1, padding="same", activation="relu")(layer)
layer = Conv2D(192, (3,3), strides=1, padding="same", activation="relu")(layer)
layer = BatchNormalization()(layer)
layer = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(layer)

layer = inceptionnet(layer, [64,  (96,128),  (16,32),  32])
layer = inceptionnet(layer, [128, (128,192), (32,96),  64])
layer = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(layer)

layer   = inceptionnet(layer, [192, (96,208),  (16,48),  64])
final_0 = helperfunction(layer, name="final_layer_0")

layer   = inceptionnet(layer, [160, (112,224), (24,64),  64])
layer   = inceptionnet(layer, [128, (128,256), (24,64),  64])
layer   = inceptionnet(layer, [112, (144,288), (32,64),  64])
final_1 = helperfunction(layer, name="final_layer_1")

layer = inceptionnet(layer, [256, (160,320), (32,128), 128])
layer = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(layer)

layer = inceptionnet(layer, [256, (160,320), (32,128), 128])
layer = inceptionnet(layer, [384, (192,384), (48,128), 128])
layer = AveragePooling2D(pool_size=(7,7), strides=1, padding="same")(layer)

layer   = Flatten()(layer)
layer   = Dropout(0.5)(layer)
layer   = Dense(256, activation="linear")(layer)
final_2 = Dense(38, activation="softmax", name="final_layer_2")(layer)

return Model(inputs=inputlayer, outputs=[final_2, final_0, final_1])
```

model = googlenet()
model.summary()

# ─── Compile ─────────────────────────────────────────────────────────────────

model.compile(
loss=[‘categorical_crossentropy’] * 3,
loss_weights=[1, 0.3, 0.3],
optimizer=Adam(learning_rate=0.001),
metrics=[‘accuracy’]
)

# ─── Callbacks ───────────────────────────────────────────────────────────────

checkpoint_path = “fine_tune_checkpoints/”
cb = [
callbacks.EarlyStopping(monitor=“val_loss”, patience=3),
callbacks.ReduceLROnPlateau(monitor=“val_loss”, factor=0.2, patience=2,
verbose=1, min_lr=1e-7),
callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True,
save_best_only=True, monitor=“val_loss”),
]

# ─── Training ────────────────────────────────────────────────────────────────

history = model.fit(
train_data,
steps_per_epoch=train_number // batch_size,
epochs=30, verbose=1, callbacks=cb,
validation_data=valid_data,
validation_steps=valid_number // batch_size,
)
model.save(“googlenet_plant_disease.h5”)

# ─── Plots ───────────────────────────────────────────────────────────────────

hist = history.history

def show_plt(metric):
plt.plot(hist[f”final_layer_2_{metric}”], label=metric)
plt.plot(hist[f”val_final_layer_2_{metric}”], label=f”val_{metric}”)
plt.ylabel(metric.capitalize()); plt.xlabel(“Epochs #”); plt.legend()
plt.show()

show_plt(“accuracy”)
show_plt(“loss”)

# ─── Evaluation ──────────────────────────────────────────────────────────────

pred_main, _, _ = model.predict(test_data)
final_predict   = np.argmax(pred_main, axis=1)
true_data       = test_data.classes

print(f’Accuracy Score: {accuracy_score(true_data, final_predict):.4f}’)
print(classification_report(true_data, final_predict, target_names=class_names))

plt.figure(figsize=(40, 40))
sns.heatmap(confusion_matrix(true_data, final_predict), annot=True, fmt=‘d’,
cmap=‘jet’, xticklabels=class_names, yticklabels=class_names, lw=6)
plt.xlabel(‘Predicted’, fontsize=20); plt.ylabel(‘True’, fontsize=20)
plt.title(‘Confusion Matrix\n’, fontsize=20); plt.show()

# ─── Single Image Prediction ─────────────────────────────────────────────────

def predict_image(image_path):
img  = Image.open(image_path).resize((120, 120))
arr  = np.array(img) / 255.0
arr  = np.expand_dims(arr, axis=0)
probs = model.predict(arr)[0]
pred_class_name = class_names[np.argmax(probs)]
max_prob = np.max(probs)
print(f’Predicted class: {pred_class_name}’)
print(f’Probability: {max_prob:.4f}’)
plt.figure(figsize=(5, 5))
plt.imshow(img); plt.axis(‘off’)
plt.text(5, 15, f’{pred_class_name}\n{max_prob:.2f}’,
fontsize=10, color=‘red’, bbox=dict(facecolor=‘white’, alpha=0.8))
plt.show()
return pred_class_name

# Example: predict_image(”/path/to/leaf.JPG”)