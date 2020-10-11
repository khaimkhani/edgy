import numpy, scipy
from scipy import special

import math
from PIL import Image


class Sobel:

    def __init__(self):

        self.h = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.v = self.h.T
        self.h_flipped = numpy.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.v_flipped = self.h_flipped.T
        self.img_width = None
        self.img_height = None

    def image2array(self, image=None):
        """
        Returns image as grayscale numpy array with zero padding.
        """
        img_arr = numpy.array(Image.open(image).convert('L'))
        self.img_width = len(img_arr[0])
        self.img_height = len(img_arr)
        return numpy.pad(img_arr, 1)

    def compute_edge_gradient(self, image=None):
        """
        Applies the vertical sobel operator to an image, f(x,y)
        """
        img = self.image2array(image)
        #print(img)
        vert_edge = self.convolve(self.v_flipped, img)
        hori_edge = self.convolve(self.h_flipped, img)
        new_boi = numpy.array([[0] * self.img_width] * self.img_height)
        #print(new_boi)
        for i in range(len(vert_edge)):
            for j in range(len(hori_edge[0])):
                # restrict between 0 - 255
                val = math.sqrt(vert_edge[i][j] ** 2 + hori_edge[i][j] ** 2)
                val /= 1052
                #val = vert_edge[i][j]
                #if val > 255:
                #    val = 255
                new_boi[i][j] = val

        #edge_img = Image.fromarray(new_boi)
        return new_boi


    def convolve(self, kernel, img):
        """
        perform convolution.
        """

        target = numpy.array([self.img_width * [0]] * self.img_height)
        for i in range(len(target)):
            for j in range(len(target[0])):
                target[i][j] = self.con_sum(kernel, img, i + 1, j + 1)
        return target

    def con_sum(self, kernel, img, i, j):
        """
        convolve at one point of image. return square of sum for gradient magnitude.
        """
        consum = 0
        for r in range(3):
            i_temp = i + r - 1
            for c in range(3):
                j_temp = j + c - 1
                consum += kernel[r][c] * img[i_temp][j_temp]
                j_temp = j - 1
            i_temp = i - 1

        return consum ** 2


if __name__ == '__main__':
    x = Sobel()
    #image = numpy.array(Image.open('grayshcale.jpeg').convert('L'))
    #x.img_width = len(image[0])
    #x.img_height = len(image)
    p = x.compute_edge_gradient('grayshcale.jpeg')

    print(p)
    i = Image.fromarray((p * 255).astype(numpy.uint8))
    i.show()
    #p.show()
    #print(x.convolve(x.h, numpy.pad(image, 1)))
    #print(image)
    #u = Image.fromarray(image)
    #u.show()
    #y = x.compute_edge_gradient()



