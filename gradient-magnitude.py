import numpy, scipy
import math
from PIL import Image


class Sobel:

    def __init__(self):
        h = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        v = h.T

    def image2array(self, image):
        """
        Returns image as grayscale numpy array with zero padding.
        :param image:
        :type image:
        :return:
        :rtype:
        """
        img_arr = numpy.array(Image.open(image).convert('L'))
        return numpy.pad(img_arr, 1)

    def compute_vert_edge(self, image):
        """
        Applies the vertical sobel operator to an image, f(x,y)
        :param image:
        :type image:
        :return:
        :rtype:
        """
        img = self.image2array(image)

    def convolve(self, img):
        """
        perform convolution. do not need to flip Sobel kernel since operations symmetric.
        :param target:
        :type target:
        :return:
        :rtype:
        """



