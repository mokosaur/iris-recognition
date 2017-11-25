import numpy as np
from segmentation import *
from coding import *
import os


def compare_codes(a, b, mask_a, mask_b, rotation=False):
    """Compares two codes and calculates Jaccard index.

    :param a: Code of the first iris
    :param b: Code of the second iris
    :param mask_a: Mask of the first iris
    :param mask_b: Mask of the second iris
    :param rotation: Maximum cyclic rotation of the code. If this argument is greater than zero, the function will
        return minimal distance of all code rotations. If this argument is False, no rotations are calculated.

    :return: Distance between two codes.
    """
    if rotation:
        d = []
        for i in range(-rotation, rotation + 1):
            c = np.roll(b, i, axis=1)
            mask_c = np.roll(mask_b, i, axis=1)
            d.append(np.sum(np.remainder(a + c, 2) * mask_a * mask_c) / np.sum(mask_a * mask_c))
        return np.min(d)
    return np.sum(np.remainder(a + b, 2) * mask_a * mask_b) / np.sum(mask_a * mask_b)


def encode_photo(image):
    """Finds the pupil and iris of the eye, and then encodes the unravelled iris.

    :param image: Image of an eye
    :return: Encoded iris (code, mask)
    :rtype: tuple (ndarray, ndarray)
    """
    img = preprocess(image)
    x, y, r = find_pupil_hough(img)
    x_iris, y_iris, r_iris = find_iris_id(img, x, y, r)
    iris = unravel_iris(image, x, y, r, x_iris, y_iris, r_iris)
    return iris_encode(iris)


def save_codes(data):
    """Takes data, and saves encoded images to 'codes' directory.

    :param data: Data formatted as returned by load_* functions from datasets.py module (dictionary with keys 'data' and
        'target')
    :type data: dict
    """
    for i in range(len(data['data'])):
        print("{}/{}".format(i, len(data['data'])))
        image = cv2.imread(data['data'][i])
        try:
            code, mask = encode_photo(image)
            np.save('codes\\code{}'.format(i), np.array(code))
            np.save('codes\\mask{}'.format(i), np.array(mask))
            np.save('codes\\target{}'.format(i), data['target'][i])
        except:
            np.save('codes\\code{}'.format(i), np.zeros(1))
            np.save('codes\\mask{}'.format(i), np.zeros(1))
            np.save('codes\\target{}'.format(i), data['target'][i])


def load_codes():
    """Loads codes saved by save_codes function.

    :return: Codes, masks, and targets of saved images
    :rtype: tuple (ndarray, ndarray, ndarray)
    """
    codes = []
    masks = []
    targets = []
    i = 0
    while os.path.isfile('codes\\code{}.npy'.format(i)):
        code = np.load('codes\\code{}.npy'.format(i))
        if code.shape[0] != 1:
            codes.append(code)
            masks.append(np.load('codes\\mask{}.npy'.format(i)))
            targets.append(np.load('codes\\target{}.npy'.format(i)))
        i += 1
    return np.array(codes), np.array(masks), np.array(targets)


def split_codes(codes, masks, targets):
    """Splits data for testing purposes.

    The first piece of data (code, mask, target) for each target is separated from the rest.

    :param codes: Array of codes
    :param masks: Array of masks
    :param targets: Array of targets
    :return: All codes, masks, and targets without the first instance of each target, then codes, masks, and targets of
        containing test examples
    :rtype: 6-tuple of ndarrays
    """
    X_test = []
    X_base = []
    M_test = []
    M_base = []
    y_test = []
    y_base = []
    for i in range(targets.max() + 1):
        X = codes[targets == i]
        M = masks[targets == i]
        X_test.append(X[0])
        X_base.append(X[1:])
        M_test.append(M[0])
        M_base.append(M[1:])
        y_test.append(i)
        y_base += [i] * X[1:].shape[0]
    return np.vstack(X_base), np.vstack(M_base), np.array(y_base), np.array(X_test), np.array(M_test), np.array(y_test)


if __name__ == '__main__':
    data = load_utiris()['data']
    image = cv2.imread(data[0])
    image2 = cv2.imread(data[6])
    print(image.shape)
    print(image2.shape)
    code, mask = encode_photo(image)
    code2, mask2 = encode_photo(image2)
    print(compare_codes(code, code2, mask, mask2))