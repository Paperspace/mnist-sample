from random import randint
import matplotlib.pyplot as plt
import requests
import tensorflow as tf


def get_random_image_from_dataset(image_index=randint(0, 9999)):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
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

    print(response.status_code)
    print(response.text)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test MNIST TF Server')
    parser.add_argument('-u', '--url', help='Prediction HOST URL', default='http://127.0.0.1:8501/v1/models/mnist:predict')
    args = parser.parse_args()

    image = get_random_image_from_dataset()
    show_selected_image(image)
    make_prediction_request(image, args.url)


if __name__ == '__main__':
    main()
