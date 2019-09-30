import os
import tensorflow as tf
from PIL import Image
from train import CNN
import numpy as np


class Attackor(object):
    def __init__(self):
        latest = tf.train.latest_checkpoint('model')
        self.cnn = CNN()
        self.cnn.model.load_weights(latest)

    def attack(self, image_path, target):
        path, name = os.path.split(image_path)
        num = os.path.splitext(name)[0].split(' - ')[1]
        # read images in gray mode
        img = Image.open(image_path)
        x = np.reshape(img, (1, 28, 28, 1)).astype(np.float32) / 255.0
        x = tf.convert_to_tensor(x, dtype=tf.float32)

        # standard target
        target = [1 if (idx == target) else 0 for idx in range(10)]
        target = np.reshape(target, (1, 10))
        target = tf.convert_to_tensor(target, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            cce = tf.keras.losses.CategoricalCrossentropy()
            y = self.cnn.model.call(x)
            loss = cce(y, target)
        gradient = tape.gradient(loss, x)
        perturb = tf.sign(gradient)

        epislons = [0.25]
        for epislon in epislons:
            perturb_x = x - perturb * epislon
            perturb_x = tf.clip_by_value(perturb_x, 0, 1)
            y = self.cnn.model.predict(x)
            before = np.argmax(y[0])
            y = self.cnn.model.predict(perturb_x)
            after = np.argmax(y[0])
            print('The filename is %s, before attack the model think it is %f, after attack the model think it is %f, epislon is %f' % (
                name, before, after, epislon))
            img = Image.fromarray((perturb_x.numpy() * 255.0).reshape(
                28, 28).astype(np.uint8), mode='L')
            img.save('perturb/%s' % (name))
        if after == 1 and after != before:
            return True
        else:
            return False


if __name__ == '__main__':
    app = Attackor()
    cnt = 0
    for name in os.listdir('img'):
        path = os.path.join('img', name)
        mark = app.attack(path, 1)
        if mark is True:
            cnt += 1

    print('access of attack is %f' % (cnt / 50))
