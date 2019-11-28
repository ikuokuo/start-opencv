#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import sys

import cv2


class Camera(object):

  def __init__(self, index=0):
    self._cap = cv2.VideoCapture(index)
    self._openni = index in (cv2.CAP_OPENNI, cv2.CAP_OPENNI2)
    self._fps = 0

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.release()

  def release(self):
    if not self._cap:
      return
    self._cap.release()
    self._cap = None

  def capture(self, cb_capture, gray=False):
    if not self._cap:
      sys.exit('The capture is not ready')

    while True:
      tick_count = cv2.getTickCount()

      if self._openni:
        if not self._cap.grab():
          sys.exit('Grabs the next frame failed')
        ret, depth = self._cap.retrieve(cv2.CAP_OPENNI_DEPTH_MAP)
        ret, frame = self._cap.retrieve(cv2.CAP_OPENNI_GRAY_IMAGE
                                        if gray else cv2.CAP_OPENNI_BGR_IMAGE)
        if cb_capture:
          cb_capture(frame, depth, self._fps)
      else:
        ret, frame = self._cap.read()
        if not ret:
          sys.exit('Reads the next frame failed')
        if gray:
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cb_capture:
          cb_capture(frame, self._fps)

      tick_count = cv2.getTickCount() - tick_count
      self._fps = cv2.getTickFrequency() / tick_count

      # esc, q
      key = cv2.waitKey(10) & 0xFF
      if key == 27 or key == ord('q'):
        break

  def fps(self):
    return self._fps

  def get(self, prop_id):
    return self._cap.get(prop_id)

  def set(self, prop_id, value):
    self._cap.set(prop_id, value)


if __name__ == '__main__':
  # callback = lambda frame, _: cv2.imshow('frame', frame)
  def callback(frame, _):
    cv2.imshow('frame', frame)

  with Camera(0) as cam:
    print('Camera: %dx%d, %d' % (
        cam.get(cv2.CAP_PROP_FRAME_WIDTH),
        cam.get(cv2.CAP_PROP_FRAME_HEIGHT),
        cam.get(cv2.CAP_PROP_FPS)))
    cam.capture(callback)

  cv2.destroyAllWindows()
