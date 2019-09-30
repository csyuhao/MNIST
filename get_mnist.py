import os
from tensorflow.keras import datasets
from PIL import Image
import numpy as np

data_path = os.path.abspath(
    os.path.dirname(__file__)) + '/data/mnist.npz'
(train_images, train_labels), (test_images,
                               test_labels) = datasets.mnist.load_data(path=data_path)
test_images = test_images.reshape((10000, 28, 28))

for idx in range(50):
    img = test_images[idx].astype(np.uint8)
    label = test_labels[idx]
    img = Image.fromarray(img, mode='L')
    img.save('img/%d - %d.jpg' % (idx, label))
