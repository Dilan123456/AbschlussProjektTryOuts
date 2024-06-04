import logging
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('C:/Users/Dilan/Documents/ML_2/AbschlussProjektTryOuts/model_fine_tuned.keras')
train_dir = 'C:/Users/Dilan/Documents/ML_2/AbschlussProjektTryOuts/images_Kopie/train'

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    seed=123,
    image_size=(256, 256),
    batch_size=16)

class_names = train_ds.class_names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    img_file = request.files['image']
    img = Image.open(img_file).convert('RGB')  # Konvertieren in RGB
    img_array = np.array(img) / 255

    img_array = (np.expand_dims(img,0))

    predictions = probability_model.predict(img_array)
    predicted_class_id = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_id]

    return jsonify({'class_name': predicted_class_name})

if __name__ == '__main__':
    app.run(debug=True)