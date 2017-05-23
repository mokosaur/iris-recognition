from datasets import load_utiris
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def show_segment(img, x, y, r, x2=None, y2=None, r2=None):
    ax = plt.subplot()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    segment = plt.Circle((x, y), r, color='b', fill=False)
    ax.add_artist(segment)
    if r2 is not None:
        segment2 = plt.Circle((x2, y2), r2, color='r', fill=False)
        ax.add_artist(segment2)
    plt.show()


def integrate(img, x0, y0, r, arc_start=0, arc_end=1, n=8):
    theta = 2 * math.pi / n
    integral = 0
    for step in range(round(arc_start * n), round(arc_end * n)):
        x = x0 + r * math.cos(step * theta)
        y = y0 + r * math.sin(step * theta)
        integral += img[round(x), round(y)]
    return integral / n


def find_segment(img, x0, y0, minr=0, maxr=500, step=1, sigma=5., center_margin=30, segment_type='pupil'):
    max_o = 0
    max_l = []
    max_x = max_y = r = 0

    bound = min(img.shape[0] - y0, img.shape[1] - x0, x0, y0)
    if maxr > bound:
        maxr = bound

    margin_img = np.pad(img[:, :, 0], maxr, 'mean')
    x0 += maxr
    y0 += maxr
    for x in range(x0 - center_margin, x0 + center_margin + 1):
        for y in range(y0 - center_margin, y0 + center_margin + 1):
            if segment_type == 'pupil':
                l = np.array([integrate(margin_img, y, x, r) for r in range(minr, maxr, step)])
            else:
                l = np.array([integrate(margin_img, y, x, r, -1 / 8, 1 / 8, n=16) +
                              integrate(margin_img, y, x, r, 3 / 8, 5 / 8, n=16)
                              for r in range(minr, maxr, step)])
            l = (l[2:] - l[:-2]) / 2
            l = gaussian_filter(l, sigma)
            l = np.abs(l)
            max_c = np.max(l)
            if max_c > max_o:
                max_o = max_c
                max_l = l
                max_x, max_y = x, y
                r = np.argmax(l) * step + minr

    return max_x - maxr, max_y - maxr, r, max_l


def _layer_to_full_image(layer):
    return np.transpose(np.array([layer, layer, layer]), (1, 2, 0))


def find_pupil_center(img):
    P = np.percentile(img[:, :, 0], 1)
    bin_pupil = np.where(img[:, :, 0] > P, 0, 255)
    kernel = np.ones((16, 16), np.uint8)
    pupil = cv2.morphologyEx(_layer_to_full_image(bin_pupil).astype('uint8'), cv2.MORPH_OPEN, kernel)
    x, y, c = 0, 0, 0
    for i in range(pupil.shape[0]):
        for j in range(pupil.shape[1]):
            if pupil[i, j, 0] > 0:
                x += i
                y += j
                c += 1
    return x / c, y / c


# Example usage
if __name__ == '__main__':
    data = load_utiris()['data']
    image = cv2.imread(data[2])
    y, x = find_pupil_center(image)
    plt.imshow(image)
    plt.scatter([x], [y])
    plt.show()
    x, y = round(x), round(y)
    print(x, y)

    # image[image > 230] = 20
    img = cv2.medianBlur(image, 17)
    x, y, r, l = find_segment(img, x, y, minr=40, maxr=150, center_margin=30, sigma=1, segment_type='pupil')
    plt.plot(range(len(l)), l)
    plt.show()
    print(x, y, r)
    img = cv2.medianBlur(image, 129)
    x2, y2, r2, l2 = find_segment(img, x, y, minr=int(1.25 * r), maxr=10 * r, center_margin=50, segment_type='iris')
    plt.plot(range(len(l2)), l2)
    plt.show()
    show_segment(image, x, y, r, x2, y2, r2)