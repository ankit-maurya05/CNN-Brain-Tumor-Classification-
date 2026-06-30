import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path)


def prepare_image(image_path, target_size=(128, 128)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)


def predict_image(model, image_path, target_size=(128, 128), class_names=None):
    image = prepare_image(image_path, target_size)
    predictions = model.predict(image)
    score = predictions[0]
    predicted_index = int(np.argmax(score))
    predicted_label = class_names[predicted_index] if class_names is not None else str(predicted_index)
    return predicted_label, score
