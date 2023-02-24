import tensorflow as tf


from flask import Flask, request
app = Flask(__name__)


@app.route('/preprocess', methods=['POST'])
def preprocess_cctv_feed():
    """
    Preprocesses a CCTV feed for use with the YOLOv5 model.
    """
    # Resize the images to the input size of the YOLOv5 model
    resized_image = tf.image.resize(cctv_feed, size=(640, 640))
    
    # Normalize the pixel values of the images
    normalized_image = tf.keras.applications.mobilenet.preprocess_input(resized_image)
    
    return normalized_image



