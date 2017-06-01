import numpy as np
from segmentation import *
from coding import *
import os


def compare_codes(a, b, mask_a, mask_b):
    return np.sum(np.remainder(a + b, 2) * mask_a * mask_b) / np.sum(mask_a * mask_b)


def encode_photo(image):
    img = preprocess(image)
    x, y, r = find_pupil_hough(img)
    x_iris, y_iris, r_iris = find_iris_id(img, x, y, r)
    iris = unravel_iris(image, x, y, r, x_iris, y_iris, r_iris)
    return iris_encode(iris)


def save_codes(data):
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