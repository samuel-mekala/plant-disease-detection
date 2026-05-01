“””
Plant Disease Detection — AlexNet
Based on SDP Report Appendix Code

Authors: Satyala Murali Karthik, Mekala Samuel, Kurmala Bhanu Prakash
VIT-AP University, December 2024
Dataset: New Plant Diseases Dataset (Kaggle) — 87,000+ images, 38 classes
“””

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

# ─── Strategy (TPU / GPU / CPU) ──────────────────────────────────────────────

try:
tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
strategy = tf.distribute.MirroredStrategy()
print(“Number of accelerators: “, strategy.num_replicas_in_sync)

# ─── Dataset Paths ───────────────────────────────────────────────────────────

# Download from Kaggle: new-plant-diseases-dataset

data_dir  = “../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)”
train_dir = data_dir + “/train”
test_dir  = data_dir + “/valid”

diseases = os.listdir(train_dir)
print(“Total disease classes are: {}”.format(len(diseases)))

# ─── Data Generators ─────────────────────────────────────────────────────────

train_datagen = ImageDataGenerator(
rescale=1./255, shear_range=0.2, zoom_range=0.2,
fill_mode=“nearest”, rotation_range=20,
width_shift_range=0.2, height_shift_range=0.2,
horizontal_flip=True, validation_split=0.2
)
test_datagen = ImageDataGenerator(
rescale=1./255, shear_range=0.2, zoom_range=0.2,
rotation_range=20, horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
train_dir, target_size=(224, 224), batch_size=128,
class_mode=‘categorical’, subset=‘training’
)
validation_set = train_datagen.flow_from_directory(
train_dir, target_size=(224, 224), batch_size=128,
class_mode=‘categorical’, subset=‘validation’, shuffle=False
)
test_set = test_datagen.flow_from_directory(
test_dir, target_size=(224, 224), batch_size=128, class_mode=‘categorical’
)

label_map = training_set.class_indices
li = list(label_map.keys())
print(“Target Classes Mapping Dict:\n”, label_map)

# ─── AlexNet Model ───────────────────────────────────────────────────────────

with strategy.scope():
model = Sequential(name=“AlexNet”)

```
model.add(Conv2D(96, (11,11), strides=(4,4), padding='valid',
                 kernel_regularizer=l2(0.0005), input_shape=(224,224,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))
model.add(BatchNormalization())

model.add(Conv2D(256, (5,5), strides=(1,1), padding='same',
                 kernel_regularizer=l2(0.0005)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))
model.add(BatchNormalization())

model.add(Conv2D(384, (3,3), strides=(1,1), padding='same',
                 kernel_regularizer=l2(0.0005)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(384, (3,3), strides=(1,1), padding='same',
                 kernel_regularizer=l2(0.0005)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(256, (3,3), strides=(1,1), padding='same',
                 kernel_regularizer=l2(0.0005)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(38, activation='softmax'))

print(model.summary())
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', TopKCategoricalAccuracy(k=1, name="top1")]
)
```

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

history = model.fit(training_set, epochs=20, verbose=1,
callbacks=cb, validation_data=validation_set)

model.load_weights(checkpoint_path)
model.evaluate(test_set)
model.save(“AlexNetModel.hdf5”)

# ─── Plots ───────────────────────────────────────────────────────────────────

hist = history.history

def show_plt(metric):
plt.plot(hist[metric],         label=metric)
plt.plot(hist[“val_” + metric],label=“val_” + metric)
plt.ylabel(metric.capitalize()); plt.xlabel(“Epochs #”); plt.legend()
plt.show()

show_plt(“accuracy”)
show_plt(“loss”)

# ─── Confusion Matrix ────────────────────────────────────────────────────────

validation_set.reset()
pred         = model.predict(validation_set)
final_predict = np.argmax(pred, axis=1)
true_data     = validation_set.classes

plt.figure(figsize=(40, 40))
sns.heatmap(confusion_matrix(true_data, final_predict), annot=True, fmt=‘d’,
cmap=‘jet’, xticklabels=li, yticklabels=li, lw=6)
plt.xlabel(‘Predicted’, fontsize=20); plt.ylabel(‘True’, fontsize=20)
plt.title(‘Confusion Matrix\n’, fontsize=20); plt.show()

print(classification_report(true_data, final_predict, target_names=li))
print(f’Accuracy Score: {accuracy_score(true_data, final_predict):.4f}’)

# ─── Single Image Prediction ─────────────────────────────────────────────────

from tensorflow.keras.preprocessing import image

def predict_image(image_path):
new_img = image.load_img(image_path, target_size=(224, 224))
img = image.img_to_array(new_img)
img = np.expand_dims(img, axis=0) / 255.0
prediction = model.predict(img).flatten()
class_name = li[prediction.argmax()]
plt.figure(figsize=(4, 4))
plt.imshow(new_img); plt.axis(‘off’); plt.title(class_name); plt.show()
return class_name

# Example: predict_image(”/path/to/leaf.JPG”)