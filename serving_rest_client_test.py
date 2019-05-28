from random import randint

plotting = True
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    print('Matplotlib not detected - images plotting not available')
    plotting = False

import numpy as np
from PIL import Image as pilimage
import requests
import tensorflow as tf

def get_image_from_drive(path):
    # Load the image
    image = pilimage.open(path)
    image = image.convert('L')
    image = np.resize(image, (28,28,1))
    image = np.array(image)
    image = image.reshape(28,28)
    return image


def get_random_image_from_dataset(image_index=randint(0, 9999)):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print('target class (from random test sample): %d' % y_test[image_index])
    return x_test[image_index]


def show_selected_image(image):
    fig = plt.figure()
    plt.subplot(1, 1, 1)
    plt.tight_layout()
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def make_vector(image):
    vector = []
    for item in image.tolist():
        vector.extend(item)
    return vector


def make_prediction_request(image, prediction_url):
    vector = make_vector(image)
    json = {
        "inputs": [vector]
    }
    response = requests.post(prediction_url, json=json)

    print('HTTP Response %s' % response.status_code)
    print(response.text)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test MNIST TF Server')
    parser.add_argument('-u', '--url', help='Prediction HOST URL', default='http://127.0.0.1:8501/v1/models/mnist:predict')
    parser.add_argument('-p', '--path', help='Example image path')
    args = parser.parse_args()
    # Load image from drive if specified, if not load example image from mnist dataset
    if args.path:
        image = get_image_from_drive(args.path)
    else:
        image = get_random_image_from_dataset()

    if plotting:
        show_selected_image(image)
    make_prediction_request(image, args.url)


if __name__ == '__main__':
    main()
