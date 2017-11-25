from datasets import load_utiris
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def show_segment(img, x, y, r, x2=None, y2=None, r2=None):
    """Shows an image with pupil and iris marked with circles.

    :param img: Image of an eye
    :param x: x coordinate of a segment
    :param y: y coordinate of a segment
    :param r: radius of a segment
    :param x2: x coordinate of another segment (optional, can be None)
    :param y2: y coordinate of another segment (optional, can be None)
    :param r2: r coordinate of another segment (optional, can be None)
    """
    ax = plt.subplot()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    segment = plt.Circle((x, y), r, color='b', fill=False)
    ax.add_artist(segment)
    if r2 is not None:
        segment2 = plt.Circle((x2, y2), r2, color='r', fill=False)
        ax.add_artist(segment2)
    plt.show()


def integrate(img, x0, y0, r, arc_start=0, arc_end=1, n=8):
    """Calculates line integral in the image.

    :param img: Image of an eye
    :param x0: x coordinate of the centre of the segment
    :param y0: y coordinate of the centre of the segment
    :param r: radius of the segment
    :param arc_start: From which point on the arc should the calculation start
    :param arc_end: At which point on the arc should the calculation end
    :param n: Number of points at which intergral is calculated along the line (the more points, the more accurate the
        result is)
    :return: Line integral value
    """
    theta = 2 * math.pi / n
    integral = 0
    for step in np.arange(arc_start * n, arc_end * n, arc_end - arc_start):
        x = int(x0 + r * math.cos(step * theta))
        y = int(y0 + r * math.sin(step * theta))
        integral += img[x, y]
    return integral / n


def find_segment(img, x0, y0, minr=0, maxr=500, step=1, sigma=5., center_margin=30, segment_type='iris', jump=1):
    """Finds the segment (pupil or iris) in the image.

    :param img: Image of an eye
    :param x0: Starting x coordinate
    :param y0: Starting y coordinate
    :param minr: Minimal radius
    :param maxr: Maximal radius
    :param step: The difference between two consecutive radii in the search space
    :param sigma: The amount of blur of integral values before selecting the optimal radius
    :param center_margin: The maximum distance from x0, y0 reached to find the optimal segment centre
    :param segment_type: Either 'iris' ot 'pupil' used to optimize the search
    :param jump: The difference between two consecutive segment centres in the search space
    :return: x, y of the segment centre, radius of the segment and integral value matching the optimal result
    """
    max_o = 0
    max_l = []

    if img.ndim > 2:
        img = img[:, :, 0]
    margin_img = np.pad(img, maxr, 'edge')
    x0 += maxr
    y0 += maxr
    for x in range(x0 - center_margin, x0 + center_margin + 1, jump):
        for y in range(y0 - center_margin, y0 + center_margin + 1, jump):
            if segment_type == 'pupil':
                l = np.array([integrate(margin_img, y, x, r) for r in range(minr, maxr, step)])
            else:
                l = np.array([integrate(margin_img, y, x, r, 1 / 8, 3 / 8, n=8) +
                              integrate(margin_img, y, x, r, 5 / 8, 7 / 8, n=8)
                              for r in range(minr + abs(x0 - x) + abs(y0 - y), maxr, step)])
            l = (l[2:] - l[:-2]) / 2
            l = gaussian_filter(l, sigma)
            l = np.abs(l)
            max_c = np.max(l)
            if max_c > max_o:
                max_o = max_c
                max_l = l
                max_x, max_y = x, y
                r = np.argmax(l) * step + minr + abs(x0 - x) + abs(y0 - y)

    return max_x - maxr, max_y - maxr, r, max_l


def _layer_to_full_image(layer):
    """Makes a full RGB image in grayscale from one layer.

    :param layer: One channel of the image
    :return: RGB image with the layer repeated in every channel
    """
    return np.transpose(np.array([layer, layer, layer]), (1, 2, 0))


def find_pupil_center(img):
    """Calculates the centre of the pupil using a naive method.

    :param img: Image of an eye
    :return: x, y coordinates of the centre of the pupil
    """
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


def preprocess(image):
    """Preprocesses the image to enhance the process of finding the iris. Crops high values of the image and blurs it.

    :param image: Image of an eye
    :return: Preprocessed image
    """
    img = image[:, :, 0].copy()
    img[img > 225] = 30
    return cv2.medianBlur(img, 21)


def find_pupil_hough(img):
    """Finds the pupil using Hugh transform.

    :param img: Image of an eye
    :return: x, y coordinates of the centre of the pupil and its radius
    """
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=10, maxRadius=200)
    circles = np.uint16(np.around(circles))
    return circles[0, 0][0], circles[0, 0][1], circles[0, 0][2]


def find_iris_id(img, x, y, r):
    """Finds the iris in the image usind integro-differential operator.

    :param img: Image of an eye
    :param x: Starting x coordinate
    :param y: Starting y coordinate
    :param r: Starting radius
    :return: x, y coordinates of the centre of the iris and its radius
    """
    x, y, r, l = find_segment(img, x, y, minr=max(int(1.25 * r), 100),
                              sigma=5, center_margin=30, jump=5)
    x, y, r, l = find_segment(img, x, y, minr=r - 10, maxr=r + 10,
                              sigma=2, center_margin=5, jump=1)
    return x, y, r


# Example usage
if __name__ == '__main__':
    data = load_utiris()['data']
    image = cv2.imread(data[0])

    img = preprocess(image)
    x, y, r = find_pupil_hough(img)
    x_iris, y_iris, r_iris = find_iris_id(img, x, y, r)
    show_segment(image, x, y, r, x_iris, y_iris, r_iris)
