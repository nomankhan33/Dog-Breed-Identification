import tensorflow as tf
import tf_keras
import tensorflow_hub as hub
import pandas as pd
import numpy as np


IMG_SIZE = 224
BATCH_SIZE = 32


def process_image(image_path, img_size=IMG_SIZE):
    """
    Takes an image path and convert it into Tensor
    """

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    return image


def get_image_label(image_path, label):

    image = process_image(image_path)
    return image, label


def create_data_batches(
    X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False
):
    """
    Creates batches of data
    shuffels the data if its taining data but does not shuffle if its validation data
    """

    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
        data_batch = data.map(process_image).batch(BATCH_SIZE)
        return data_batch

    if valid_data:
        print("Creating validating data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
        return data_batch

    else:
        print("Creating training data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data = data.shuffle(buffer_size=len(X))
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
        return data_batch


def load_model(model_path):

    print(f"Loading saved model from {model_path}...")
    model = tf_keras.models.load_model(
        model_path, custom_objects={"KerasLayer": hub.KerasLayer}
    )
    return model


labels_csv = pd.read_csv("labels.csv")
labels = labels_csv["breed"]
labels = np.array(labels)
unique_breeds = np.unique(labels)


def get_pred_label(prediction_probabilities):
    """
    Turns an array of prection of labels
    """
    return unique_breeds[np.argmax(prediction_probabilities)]