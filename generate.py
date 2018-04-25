import random

import numpy as np
from PIL import Image


def __debug_save_image(array, cls, idx):
    img = Image.new('RGB', (28, 28))
    img.putdata([(int(i * 255), int(i * 255), int(i * 255)) for i in array])
    img.save('data/test/%d/%d.png' % (idx // 10000, idx))


dataset_train = np.load('data/test.npy')

d = {}
for i in range(dataset_train.shape[0]):
    __debug_save_image(dataset_train[i, :784], None, i)

for key in sorted(d.keys()):
    print(key, d[key])
