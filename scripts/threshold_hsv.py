#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring

# Finding red color in image using Python & OpenCV
#  https://stackoverflow.com/questions/30331944/finding-red-color-in-image-using-python-opencv

import sys

import cv2
import numpy as np

if len(sys.argv) <= 1:
  sys.exit(f'python {sys.argv[0]} <image>')

# Load in image
image = cv2.imread(sys.argv[1])

# Create a window
cv2.namedWindow('image')

def nothing(x):
  pass

# create trackbars for color change
cv2.createTrackbar('HMin', 'image', 0, 179, nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

output = image
wait_time = 33

while True:
  # get current positions of all trackbars
  hMin = cv2.getTrackbarPos('HMin','image')
  sMin = cv2.getTrackbarPos('SMin','image')
  vMin = cv2.getTrackbarPos('VMin','image')

  hMax = cv2.getTrackbarPos('HMax','image')
  sMax = cv2.getTrackbarPos('SMax','image')
  vMax = cv2.getTrackbarPos('VMax','image')

  # Set minimum and max HSV values to display
  lower = np.array([hMin, sMin, vMin])
  upper = np.array([hMax, sMax, vMax])

  # Create HSV Image and threshold into a range.
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv, lower, upper)
  output = cv2.bitwise_and(image,image, mask= mask)

  # Print if there is a change in HSV value
  if (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax):
    print(f"(hMin={hMin}, sMin={sMin}, vMin={vMin}), (hMax={hMax}, sMax={sMax}, vMax={vMax})")
    phMin = hMin
    psMin = sMin
    pvMin = vMin
    phMax = hMax
    psMax = sMax
    pvMax = vMax

  # Display output image
  cv2.imshow('image', output)

  # Wait longer to prevent freeze for videos.
  if cv2.waitKey(wait_time) & 0xFF == ord('q'):
    break

cv2.destroyAllWindows()
