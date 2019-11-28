#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring,import-error,wrong-import-order
import cv2
import numpy as np

from common.camera import Camera


def main():
  def callback(frame, _):
    edges = cv2.Canny(frame, 100, 200)
    cv2.imshow('frame', np.vstack((frame, edges)))

  with Camera(0) as cam:
    print('Camera: %dx%d, %d' % (
        cam.get(cv2.CAP_PROP_FRAME_WIDTH),
        cam.get(cv2.CAP_PROP_FRAME_HEIGHT),
        cam.get(cv2.CAP_PROP_FPS)))
    cam.capture(callback, True)

  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()


# Canny Edge Detection:
#   http://docs.opencv.org/master/da/d22/tutorial_py_canny.html
