#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import sys
import time

import cv2 as cv
import numpy as np


def _parse_args():
  import argparse
  parser = argparse.ArgumentParser(
      usage="python src/python/stitching/overlap_finding.py data/stitching/newspaper*.jpg -p --rtl")

  parser.add_argument("img_names", nargs="+", type=str, help="files to stitch")
  parser.add_argument("-p", "--preview", dest="preview", action="store_true",
      help="Run stitching in the preview mode")
  parser.add_argument("--work_megapix", dest="work_megapix", default=-1, type=float,
      help="Resolution for image registration step. The default is %(default)s Mpx")
  parser.add_argument("-f", "--features", dest="features", default="sift", type=str,
      help="Type of features used for images matching. The default is %(default)s")

  parser.add_argument("-m", "--matcher", dest="matcher", default="bf", type=str,
      help="Matcher used for pairwise image matching. The choices are 'bf' or 'flann'")
  parser.add_argument("--good_coeff", dest="good_coeff", default=0.3, type=float,
      help="Coefficient for ratio test to find good matches")
  parser.add_argument("--rtl", dest="rtl", action="store_true",
      help="Images from right to left")

  args = parser.parse_args()

  print("Args")
  print("  img_names: {}".format(args.img_names))
  print("  preview: {}".format(args.preview))
  print("  work_megapix: {}".format(args.work_megapix))
  print("  features: {}".format(args.features))

  print("  matcher: {}".format(args.matcher))
  print("  good_coeff: {}".format(args.good_coeff))
  print("  rtl: {}".format(args.rtl))

  return args


def _main():
  args = _parse_args()
  img_names = args.img_names

  if not img_names:
    sys.exit("python overlap_finding.py boat*.jpg")

  features_type = args.features
  if features_type == "orb":
    finder = cv.ORB.create()
  elif features_type == "surf":
    finder = cv.xfeatures2d_SURF.create()
  elif features_type == "sift":
    finder = cv.SIFT.create()
    # finder = cv.xfeatures2d_SIFT.create()
  else:
    sys.exit("Unknown features type")

  preview = args.preview

  work_megapix = args.work_megapix
  is_work_scale_set = False

  full_img_sizes = []
  features = []
  images = []

  print()
  for idx, name in enumerate(img_names):
    full_img = cv.imread(cv.samples.findFile(name))
    if full_img is None:
      sys.exit(f"Cannot read image {name}")

    full_img_sizes.append((full_img.shape[1], full_img.shape[0]))

    if work_megapix < 0:
      img = full_img
      work_scale = 1
      is_work_scale_set = True
    else:
      if is_work_scale_set is False:
        work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0]*full_img.shape[1])))
        is_work_scale_set = True
      img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)

    t = time.time()

    img_fea = cv.detail.computeImageFeatures2(finder, img)
    img_fea.img_idx = idx
    features.append(img_fea)

    keypoints = img_fea.getKeypoints()
    print(f"{name} {img.shape[1]}x{img.shape[0]} {work_scale:.2f}, "
          f"keypoints: {len(keypoints)}, cost: {time.time()-t:.2f} s")

    images.append(img)

    if preview:
      img = cv.drawKeypoints(img, keypoints, None)

      cv.imshow(f"{name} {img.shape[1]}x{img.shape[0]}", img)
      cv.waitKey(0)

  cv.destroyAllWindows()

  num_images = len(img_names)
  if num_images < 2:
    sys.exit("Need more images")

  # overlap finding

  # Feature Matching
  #  https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
  # Geometric Transformations of Images
  #  https://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html
  # BestOf2NearestMatcher
  #  opencv/modules/stitching/src/matchers.cpp
  # https://github.com/Mostafa-Shabani/Find_Overlap_OpenCV/blob/master/Overlap.cpp
  # https://medium.com/analytics-vidhya/image-stitching-with-opencv-and-python-1ebd9e0a6d78

  matcher_type = args.matcher
  if matcher_type == "bf":
    matcher = cv.BFMatcher()
  elif matcher_type == "flann":
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv.FlannBasedMatcher(index_params, search_params)
  else:
    sys.exit("Unknown matcher type")

  good_coeff = args.good_coeff
  rtl = args.rtl

  print()
  for i in range(0, num_images-1):
    feat1, feat2 = features[i:i+2]
    kp1, des1 = feat1.getKeypoints(), feat1.descriptors
    kp2, des2 = feat2.getKeypoints(), feat2.descriptors
    if rtl:
      _overlap2(i, images[i], kp1, des1, i+1, images[i+1], kp2, des2, matcher,
          good_coeff=good_coeff, stitching=True, preview=preview)
    else:
      _overlap2(i+1, images[i+1], kp2, des2, i, images[i], kp1, des1, matcher,
          good_coeff=good_coeff, stitching=True, preview=preview)


def _overlap(i1, img1, i2, img2, finder, matcher,
    good_coeff=0.3, stitching=False, preview=True):
  img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
  img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

  kp1, des1 = finder.detectAndCompute(img1_gray, None)
  kp2, des2 = finder.detectAndCompute(img2_gray, None)

  _overlap2(i1, img1, kp1, des1, i2, img2, kp2, des2, matcher,
      good_coeff=good_coeff, stitching=stitching, preview=preview)


def _overlap2(i1, img1, kp1, des1, i2, img2, kp2, des2, matcher,
    good_coeff=0.3, stitching=False, preview=True):
  matches = matcher.knnMatch(des1, des2, k=2)

  good = []
  for m, n in matches:
    if m.distance < good_coeff * n.distance:
      good.append(m)
  print(f"Good matches: {len(good)}")

  if preview:
    draw_params = dict(matchColor=(0, 255, 0), flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    cv.imshow(f"match {i1}>{i2} {img3.shape[1]}x{img3.shape[0]}", img3)
    cv.waitKey(0)

  MIN_MATCH_COUNT = 10
  if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    if preview:
      h, w = img1.shape[:2]
      pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
      dst = np.int32(cv.perspectiveTransform(pts, M))
      img3 = cv.polylines(img2.copy(), [dst], True, 255, 3, cv.LINE_AA)
      cv.imshow("overlapping", img3)
      cv.waitKey(0)
  else:
    print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
    return

  if stitching:
    dst = cv.warpPerspective(img1, M, (img1.shape[1]+img2.shape[1], img1.shape[0]))
    dst[0:img2.shape[0], 0:img2.shape[1]] = img2
  else:
    dst = cv.warpPerspective(img1, M, (img1.shape[1], img1.shape[0]))
  if preview:
    cv.imshow("stitching.jpg", _crop(dst))
    cv.waitKey(0)

  if preview:
    cv.destroyAllWindows()


# Crop black edges with OpenCV
#  https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
def _crop(img):
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
  contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  x, y, w, h = cv.boundingRect(contours[0])
  return img[y:y+h, x:x+w]


if __name__ == "__main__":
  _main()
