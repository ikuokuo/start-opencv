#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring,import-error,wrong-import-order,invalid-name
import cv2
import numpy as np

from common.camera import Camera
from common.draw import Gravity, put_text


def stereo2depth(stereo, gray_l, gray_r, fps):
  # Compute the disparity image
  disp = stereo.compute(gray_l, gray_r)

  # Normalize the image for representation
  min_, max_ = disp.min(), disp.max()
  disp = np.uint8(255 * (disp - min_) / (max_ - min_))

  # Resolution
  put_text(gray_l, '{1}x{0}'.format(*gray_l.shape), Gravity.TOP_LEFT)
  put_text(disp, '{1}x{0}'.format(*disp.shape), Gravity.TOP_LEFT)
  # FPS
  put_text(disp, '%.1f' % fps, Gravity.TOP_RIGHT)

  # Display the result
  cv2.imshow('disparity', np.hstack((gray_l, disp)))
  # cv2.imshow('gray_l', gray_l)
  # cv2.imshow('disparity', disp)


def stereo2depth_whole(stereo, gray, fps):
  if len(gray.shape) > 2 and gray.shape[2] > 1:
    # cv2.imshow('color', gray)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
  h, w = gray.shape[:2]
  v = w / 2
  stereo2depth(stereo, gray[0:h, 0:v], gray[0:h, v:w], fps)


def main():
  # Initialize the stereo block matching object
  stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

  import functools
  callback = functools.partial(stereo2depth_whole, stereo)

  with Camera(0) as cam:
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1344)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 376)
    print('Camera: %dx%d, %d' % (
        cam.get(cv2.CAP_PROP_FRAME_WIDTH),
        cam.get(cv2.CAP_PROP_FRAME_HEIGHT),
        cam.get(cv2.CAP_PROP_FPS)))
    cam.capture(callback, False)

  cv2.destroyAllWindows()


def test():
  stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

  gray_l = cv2.imread('data/calib3d/tsukuba_l.png', 0)
  gray_r = cv2.imread('data/calib3d/tsukuba_r.png', 0)

  stereo2depth(stereo, gray_l, gray_r, 0)

  cv2.waitKey()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  # main()
  test()


# Depth Map from Stereo Images:
#   http://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html
