from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# with open("ricejson.json", "r") as json_file:
#     model_json = json_file.read()
# model = tf.keras.models.model_from_json(model_json)

# Cargar el modelo de TensorFlow
model = tf.keras.models.load_model('modelo_completo_arroz_final.h5')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['image']
    img = Image.open(file).convert('RGB')

    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    return jsonify({
        "class": int(predicted_class),
        "confidence": float(confidence)
    })


if __name__ == '__main__':
    app.run(debug=True)
