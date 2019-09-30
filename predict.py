import os
import tensorflow as tf
from PIL import Image
import numpy as np
from train import CNN


class Predictor(object):
    def __init__(self):
        latest = tf.train.latest_checkpoint('model')
        self.cnn = CNN()
        # restore model parameters
        self.cnn.model.load_weights(latest)

    def predict(self, image_path):
        # read images in gray mode
        img = Image.open(image_path).convert('L')
        flatten_img = np.reshape(img, (1, 28, 28, 1)) / 255.0

        y = self.cnn.model.predict(flatten_img)

        path, name = os.path.split(image_path)
        num = os.path.splitext(name)[0].split(' - ')[1]
        print('file name is %s, digital number is %s, predicted number is %d' %
              (name, num, np.argmax(y[0])))


if __name__ == '__main__':
    app = Predictor()

    for name in os.listdir('img'):
        path = os.path.join('img', name)
        app.predict(path)
