import math
import numpy as np

from skimage.util import view_as_blocks


def polar2cart(r, x0, y0, theta):
    x = int(x0 + r * math.cos(theta))
    y = int(y0 + r * math.sin(theta))
    return x, y


def unravel_iris(img, xp, yp, rp, xi, yi, ri, phase_width=300, iris_width=150):
    if img.ndim > 2:
        img = img[:, :, 0].copy()
    iris = np.zeros((iris_width, phase_width))
    theta = np.linspace(0, 2 * np.pi, phase_width)
    for i in range(phase_width):
        begin = polar2cart(rp, xp, yp, theta[i])
        end = polar2cart(ri, xi, yi, theta[i])
        xspace = np.linspace(begin[0], end[0], iris_width)
        yspace = np.linspace(begin[1], end[1], iris_width)
        iris[:, i] = [255 - img[int(y), int(x)]
                      if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]
                      else 0
                      for x, y in zip(xspace, yspace)]
    return iris


def gabor(rho, phi, w, theta0, r0, alpha, beta):
    return np.exp(-w * 1j * (theta0 - phi)) * np.exp(-(rho - r0) ** 2 / alpha ** 2) * \
           np.exp(-(phi - theta0) ** 2 / beta ** 2)


def gabor_convolve(img, w, alpha, beta):
    rho = np.array([np.linspace(0, 1, img.shape[0]) for i in range(img.shape[1])]).T
    x = np.linspace(0, 1, img.shape[0])
    y = np.linspace(-np.pi, np.pi, img.shape[1])
    xx, yy = np.meshgrid(x, y)
    return rho * img * np.real(gabor(xx, yy, w, 0, 0.5, alpha, beta).T), \
           rho * img * np.imag(gabor(xx, yy, w, 0, 0.5, alpha, beta).T)


def iris_encode(img, dr=15, dtheta=15, alpha=0.4):
    # mean = np.mean(img)
    # std = img.std()
    mask = view_as_blocks(np.logical_and(100 < img, img < 230), (dr, dtheta))
    norm_iris = (img - img.mean()) / img.std()
    patches = view_as_blocks(norm_iris, (dr, dtheta))
    code = np.zeros((patches.shape[0] * 3, patches.shape[1] * 2))
    code_mask = np.zeros((patches.shape[0] * 3, patches.shape[1] * 2))
    for i, row in enumerate(patches):
        for j, p in enumerate(row):
            for k, w in enumerate([8, 16, 32]):
                wavelet = gabor_convolve(p, w, alpha, 1 / alpha)
                code[3 * i + k, 2 * j] = np.sum(wavelet[0])
                code[3 * i + k, 2 * j + 1] = np.sum(wavelet[1])
                code_mask[3 * i + k, 2 * j] = code_mask[3 * i + k, 2 * j + 1] = \
                    1 if mask[i, j].sum() > dr * dtheta * 3 / 4 else 0
    code[code >= 0] = 1
    code[code < 0] = 0
    return code, code_mask


if __name__ == '__main__':
    import cv2
    from datasets import load_utiris
    import matplotlib.pyplot as plt

    data = load_utiris()['data']
    image = cv2.imread(data[0])

    iris = unravel_iris(image, 444, 334, 66, 450, 352, 245)
    code, mask = iris_encode(iris)

    plt.subplot(211)
    plt.imshow(iris, cmap=plt.cm.gray)
    plt.subplot(223)
    plt.imshow(code, cmap=plt.cm.gray, interpolation='none')
    plt.subplot(224)
    plt.imshow(mask, cmap=plt.cm.gray, interpolation='none')
    plt.show()
