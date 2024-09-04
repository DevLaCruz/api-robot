
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

model = tf.keras.models.load_model('modelo_completo_arroz_final.h5')

model_json = model.to_json()
with open("ricejson.json", "w") as json_file:
    json_file.write(model_json)
