#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import argparse
import os
import sys

import cv2 as cv
import numpy as np


def _contours(threshold, show_contours=True, show_boxes=True,
    show_circles=True):
  contours, hierarchy = cv.findContours(
    threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

  # approximates a polygonal curve(s) with the specified precision
  contours_poly = [cv.approxPolyDP(c, 3, True) for c in contours]

  h, w = threshold.shape[:2]
  drawing = np.zeros((h, w, 3), dtype=np.uint8)

  color = (0, 255, 0)
  thickness = 2
  line_type = cv.LINE_8

  # contours
  if show_contours:
    for i in range(len(contours)):
      cv.drawContours(drawing, contours_poly, i, color, 1, line_type,
                      hierarchy)

  # bounding boxes
  if show_boxes:
    for contour in contours_poly:
      rect = cv.boundingRect(contour)
      cv.rectangle(drawing,
                   (int(rect[0]), int(rect[1])),
                   (int(rect[0]+rect[2]), int(rect[1]+rect[3])),
                   color, thickness, line_type)

  # bounding circles
  if show_circles:
    for contour in contours_poly:
      center, radius = cv.minEnclosingCircle(contour)
      cv.circle(drawing, (int(center[0]), int(center[1])), int(radius),
                color, thickness, line_type)

  # show
  if show_contours or show_boxes or show_circles:
    cv.imshow("contours", drawing)
    return drawing
  else:
    return None


def _threshold(image):
  if len(image.shape) == 2:
    gray = image
  else:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  _, thres = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
  return thres


def _threshold_hsv(image, lower, upper):
  hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
  mask = cv.inRange(hsv, lower, upper)
  result = cv.bitwise_and(image, image, mask=mask)
  return result, mask


def _kernel(kernel_size, kernel_type=0):
  if kernel_type == 0:
    kernel_type = cv.MORPH_RECT
  elif kernel_type == 1:
    kernel_type = cv.MORPH_CROSS
  elif kernel_type == 2:
    kernel_type = cv.MORPH_ELLIPSE
  return cv.getStructuringElement(kernel_type,
    (2*kernel_size + 1, 2*kernel_size + 1), (kernel_size, kernel_size))


def _morph(thres, op, kernel):
  return cv.morphologyEx(thres, op, kernel)


def _rand_color():
  from random import randint
  return (randint(0,256), randint(0,256), randint(0,256))


def _parse_args():
  def ints_type(string, num=3, sep=","):
    if not string:
      return None
    ints = string.split(sep)
    ints_len = len(ints)
    if ints_len != num:
      sys.exit(f"error: ints_type size must be {num}")
    return [int(x) for x in ints]

  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--image", required=True, help="the image path")

  parser.add_argument("-to", "--thres-on", action="store_true",
    help="not thresholding, read image grayscale instead of color")

  parser.add_argument("--thres-hsv", action="store_true",
    help="use threshold hsv")
  parser.add_argument("--thres-hsv-lower", type=ints_type, default=[0,110,190],
    help="the threshold hsv lower values: %(default)s")
  parser.add_argument("--thres-hsv-upper", type=ints_type, default=[7,255,255],
    help="the threshold hsv upper values: %(default)s")
  parser.add_argument("--thres-hsv-", action="store_true",
    help="use threshold hsv")

  parser.add_argument("-mo", "--morph-open", action="store_true",
    help="morph open")
  parser.add_argument("-mc", "--morph-close", action="store_true",
    help="morph close")
  parser.add_argument("-ks", "--kernel-size", type=int, default=1,
    help="kernel size, default: %(default)s")
  parser.add_argument("-kt", "--kernel-type", type=int, default=0,
    help="kernel type, 0: RECT, 1: CROSS, 2: ELLIPSE")

  parser.add_argument("--no-show-contours", action="store_true")
  parser.add_argument("--no-show-boxes", action="store_true")
  parser.add_argument("--no-show-circles", action="store_true")

  args = parser.parse_args()

  print("Args")
  print(f"  image: {args.image}")
  print(f"  thres_on: {args.thres_on}")
  print(f"  thres_hsv: {args.thres_hsv}")
  print(f"  thres_hsv_lower: {args.thres_hsv_lower}")
  print(f"  thres_hsv_upper: {args.thres_hsv_upper}")
  print(f"  morph_open: {args.morph_open}")
  print(f"  morph_close: {args.morph_close}")
  print(f"  kernel_size: {args.kernel_size}")
  print(f"  kernel_type: {args.kernel_type}")
  print(f"  no_show_contours: {args.no_show_contours}")
  print(f"  no_show_boxes: {args.no_show_boxes}")
  print(f"  no_show_circles: {args.no_show_circles}")
  return args


def _main():
  args = _parse_args()

  if args.thres_on:
    img = cv.imread(args.image, cv.IMREAD_COLOR)
  else:
    img = cv.imread(args.image, cv.IMREAD_GRAYSCALE)

  cv.imshow("image", img)
  cv.waitKey(500)

  if args.thres_on:
    if args.thres_hsv:
      _, thres = _threshold_hsv(img, np.array(args.thres_hsv_lower),
                                np.array(args.thres_hsv_upper))
    else:
      thres = _threshold(img)
  else:
    thres = img

  if args.morph_open or args.morph_close:
    kernel = _kernel(args.kernel_size, args.kernel_type)
    if args.morph_open:
      thres = _morph(thres, cv.MORPH_OPEN, kernel)
    if args.morph_close:
      thres = _morph(thres, cv.MORPH_CLOSE, kernel)

  cv.imshow("thres", thres)
  cv.waitKey(500)

  drawing = _contours(thres,
                      show_contours=not args.no_show_contours,
                      show_boxes=not args.no_show_boxes,
                      show_circles=not args.no_show_circles)
  if drawing is not None:
    root, ext = os.path.splitext(args.image)
    save_path = f"{root}_contours{ext}"
    cv.imwrite(save_path, drawing)
    print(f"Contours output: {save_path}")

  cv.waitKey()


if __name__ == "__main__":
  _main()
