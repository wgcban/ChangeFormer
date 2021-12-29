import os
import numpy as np
from PIL import Image


if __name__ == '__main__':
    filepath = r"/media/lidan/ssd2/CDData/LEVIR-CD256/B/"  # Dataset directory
    pathDir = os.listdir(filepath)  # Images in dataset directory
    num = len(pathDir)  # Here (512512) is the size of each image

    print("Computing mean...")
    data_mean = np.zeros(3)
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = Image.open(os.path.join(filepath, filename))
        img = np.array(img) / 255.0
        print(img.shape)
        data_mean += np.mean(img)  # Take all the data of the first dimension in the three-dimensional matrix
		# As the use of gray images, so calculate a channel on it
    data_mean = data_mean / num

    print("Computing var...")
    data_std = 0.
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = Image.open(os.path.join(filepath, filename)).convert('L').resize((256, 256))
        img = np.array(img) / 255.0
        data_std += np.std(img)

    data_std = data_std / num
    print("mean:{}".format(data_mean))
    print("std:{}".format(data_std))