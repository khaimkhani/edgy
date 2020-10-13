import numpy
from collections import deque
from filter import *
from PIL import Image


class Sobel:

    def __init__(self):

        self.h = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.v = self.h.T

        self.h_flipped = numpy.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.v_flipped = self.h_flipped.T
        self.img_width = None
        self.img_height = None
        self.custom = None

    def use_gauss(self, k_width, sigma):
        """
        Override Sobel filter to use Gaussian filter instead.
        """
        arr = gaussian_subfilter(k_width, sigma)
        sum = 0
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                sum += arr[i][j]

        arr /= sum
        self.custom = form_kernel(arr, arr)


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

        #img = self.image2array(image)

        vert_edge = self.convolve(self.v_flipped, image)
        hori_edge = self.convolve(self.h_flipped, image)

        new_boi = numpy.array([[0] * self.img_width] * self.img_height)

        for i in range(len(vert_edge)):
            for j in range(len(hori_edge[0])):
                # restrict between 0 - 255
                val = (vert_edge[i][j] ** 2 + hori_edge[i][j] ** 2) ** 0.5



                #    val = 255
                new_boi[i][j] = val
        print(new_boi)
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
        convolve at one point of image. return sum for gradient magnitude.
        """
        consum = 0
        for r in range(len(kernel)):
            i_temp = i + r - 1
            for c in range(len(kernel[0])):
                j_temp = j + c - 1
                consum += kernel[r][c] * img[i_temp][j_temp]
                j_temp = j - 1
            i_temp = i - 1

        return consum

    def threshold(self, img):
        """
        threshold calculator.
        """
        sum = 0
        for i in range(len(img)):
            for j in range(len(img[0])):
                sum += img[i][j]
        thres = sum / (len(img) * len(img[0]))

        new_thres = self.thresh_rec(thres, img)
        for i in range(len(img)):
            for j in range(len(img[0])):
                if (img[i][j]) < new_thres:
                    img[i][j] = 0
                else:
                    img[i][j] = 255
        return img

    def thresh_rec(self, thres, img):
        """
        helper function.
        """
        prev_thres = 0
        curr_thres = thres
        while abs(prev_thres - curr_thres) > 0.01:
            print(curr_thres)
            sum_l = 0
            l = 0
            sum_h = 0
            h = 0

            for i in range(len(img)):
                for j in range(len(img[0])):
                    if (img[i][j]) < curr_thres:

                        sum_l += img[i][j]
                        l += 1
                    else:
                        sum_h += img[i][j]
                        h += 1
            prev_thres = curr_thres
            curr_thres = ((sum_l / l) + (sum_h / h)) / 2

        return curr_thres

    def ccl_re(self, img):
        """
        connected component labelling
        """
        connected = numpy.array([[0] * self.img_width] * self.img_height)
        que = deque()
        it = 0
        for i in range(len(img)):
            for j in range(len(img[0])):
                if img[i][j] == 255 and connected[i][j] == 0:
                    it += 1
                    connected[i][j] = it
                    que.appendleft((i, j))

                while len(que) > 0:
                    t = que.pop()

                    for n in [-1, 0, 1]:
                        for m in [-1, 0, 1]:
                            if (t[0] + n >= 0 and t[1] + m >= 0) and (t[0] + n < self.img_height and t[1] + m < self.img_width):
                                if img[t[0] + n][t[1] + m] == 255 and connected[t[0] + n][t[1] + m] == 0:
                                    connected[t[0] + n][t[1] + m] = it
                                    que.appendleft((t[0] + n, t[1] + m))

        return it


if __name__ == '__main__':

    x = Sobel()
    image = numpy.array(Image.open('Q6.png').convert('L'))
    x.img_width = len(image[0])
    x.img_height = len(image)
    image = numpy.pad(image, 1)
    x.use_gauss(1, 0.7)
    #print(x.h_flipped)
    imgse = x.convolve(x.custom, image)

    imgse = numpy.pad(imgse, 1)
    p = x.compute_edge_gradient(imgse)
    #img = numpy.array(Image.open('Q6.png').convert('L'))
    #x.img_height = len(img)
    #x.img_width = len(img[0])
    #img = numpy.pad(img, 1)
    #p = x.convolve(x.h_flipped, img)
    x.threshold(p)
    #print(p)
    pic = p.astype(numpy.uint8)
    print(pic)
    print(x.ccl_re(pic))
    i = Image.fromarray(pic)

    i.show()

    #p.show()
    #print(x.convolve(x.h, numpy.pad(image, 1)))
    #print(image)
    #u = Image.fromarray(image)
    #u.show()
    #y = x.compute_edge_gradient()



