import numpy, scipy
import math


def gaussian_subfilter(k_width, sigma):
    width = (2*k_width) + 1
    mu = k_width + 1

    rel_weights = []
    for i in range(width):
        val = round(gaus(i + 1, mu, sigma), 1)
        if val == 0:
            val = 0.1
        rel_weights.append(val)

    subkernel = numpy.array([rel_weights]) / min(rel_weights)

    return subkernel


def gaus(x, mu, sigma):
    gaus = (1 / (sigma * (2 * math.pi) ** 0.5)) * (math.exp((((x - mu)**2) / (sigma**2 * 2)) * -0.5))
    return gaus


def form_kernel(array_1, array_2):
    return array_1.T * array_2


if __name__ == '__main__':
    print(gaussian_subfilter(1, 0.7))
