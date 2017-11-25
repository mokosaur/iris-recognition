import os
import numpy as np


def load_utiris():
    """Fetches NIR images from UTIRIS dataset.

    Retrieves image paths and labels for each NIR image in the dataset. There should already exist a directory named
    'UTIRIS V.1'. If it does not exist then download the dataset from the official page (https://utiris.wordpress.com/).

    :return: A dictionary with two keys: 'data' contains all images paths, 'target' contains the image labels - each eye
        gets its unique number.
    """
    data = []
    target = []
    target_i = 0
    index_used = False
    for dirpath, dirnames, filenames in os.walk('UTIRIS V.1\\Infrared Images'):
        for f in filenames:
            if f.endswith('.bmp'):
                data.append('{}\{}'.format(dirpath, f))
                target.append(target_i)
                index_used = True
        if index_used:
            target_i += 1
            index_used = False
    return {'data': np.array(data),
            'target': np.array(target)}


# Example usage
if __name__ == '__main__':
    import cv2

    data = load_utiris()['data']
    image = cv2.imread(data[0])
    cv2.imshow('test', image)
    cv2.waitKey(0)
