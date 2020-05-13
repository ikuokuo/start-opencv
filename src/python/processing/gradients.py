#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import cv2 as cv

from matplotlib import pyplot as plt


def _gradients(path):
  img = cv.imread(path, 0)

  laplacian = cv.Laplacian(img, cv.CV_64F)
  sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
  sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

  plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
  plt.title('Original'), plt.xticks([]), plt.yticks([])

  plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
  plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

  plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
  plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

  plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
  plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

  plt.show()


if __name__ == "__main__":
  import sys
  if len(sys.argv) <= 1:
    sys.exit("python gradients.py <image>")
  _gradients(sys.argv[1])
