import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

new_model = tf.keras.models.load_model('model_fine_tuned.keras')

test_dir = 'C:/Users/Dilan/Documents/ML_2/AbschlussProjektTryOuts/images_Kopie/test'

batch_size = 16
img_height = 256
img_width = 256

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

loss, acc = new_model.evaluate(test_ds, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_ds).shape)
