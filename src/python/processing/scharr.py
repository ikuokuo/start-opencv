#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import sys

import cv2 as cv


def main(argv):
  window_name = ('Sobel Demo - Simple Edge Detector')
  scale = 1
  delta = 0
  ddepth = cv.CV_16S

  if len(argv) < 1:
    print('Not enough parameters')
    print('Usage:\nmorph_lines_detection.py < path_to_image >')
    return -1

  # Load the image
  src = cv.imread(argv[0], cv.IMREAD_COLOR)

  # Check if image is loaded fine
  if src is None:
    print('Error opening image: ' + argv[0])
    return -1

  src = cv.GaussianBlur(src, (3, 3), 0)

  gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

  grad_x = cv.Scharr(gray, ddepth, 1, 0)
  # Gradient-Y
  grad_y = cv.Scharr(gray, ddepth, 0, 1)

  abs_grad_x = cv.convertScaleAbs(grad_x)
  abs_grad_y = cv.convertScaleAbs(grad_y)

  grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

  cv.imshow(window_name, grad)
  cv.waitKey(0)

  return 0


if __name__ == "__main__":
  main(sys.argv[1:])
