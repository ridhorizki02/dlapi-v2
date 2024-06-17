import tensorflow as tf
import os

model = tf.saved_model.load('app/utils/breacnet')
labels_file = os.path.join('app/utils/labels.txt')

def load_labels():
    with open(labels_file, 'r') as file:
        labels = [line.strip() for line in file]
    return labels

def predict_thermal_image(preprocessed_image):
    output = model.signatures["serving_default"](tf.constant(preprocessed_image))
    predicted_class = tf.argmax(output["softmax"], axis=1)[0]
    probabilities = tf.nn.softmax(output["softmax"], axis=1)
    predicted_prob = probabilities[0, predicted_class]
    labels = load_labels()
    predicted_label = labels[predicted_class.numpy()]
    return predicted_label, predicted_prob.numpy() * 100
