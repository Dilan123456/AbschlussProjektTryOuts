import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Laden des Datensatzes
train_dir = 'C:/Users/Dilan/Documents/ML_2/AbschlussProjektTryOuts/images_Kopie/train'
validation_dir = 'C:/Users/Dilan/Documents/ML_2/AbschlussProjektTryOuts/images_Kopie/validation'

batch_size = 16
img_height = 256
img_width = 256

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

# Datapreparation for the Benchmark-Model (RandomForest)
def extract_features_labels(dataset):
    features = []
    labels = []
    for images, label_batch in dataset:
        for img, lbl in zip(images, label_batch):
            features.append(img.numpy().flatten())
            labels.append(lbl.numpy())
    return np.array(features), np.array(labels)

# Extract Features and Labels
train_features, train_labels = extract_features_labels(train_ds)
val_features, val_labels = extract_features_labels(val_ds)

# Benchmark-Model: RandomForest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(train_features, train_labels)

# Prediction and evaluation
val_predictions_rf = rf_model.predict(val_features)

print('Random Forest Classification Report')
print(classification_report(val_labels, val_predictions_rf))
print('Random Forest Confusion Matrix')
print(confusion_matrix(val_labels, val_predictions_rf))
print('Random Forest Accuracy')
print(accuracy_score(val_labels, val_predictions_rf))



