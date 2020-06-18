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
      usage="python src/python/stitching/features_finding.py data/stitching/boat*.jpg -p")

  parser.add_argument("img_names", nargs="+", type=str, help="files to stitch")
  parser.add_argument("-p", "--preview", dest="preview", action="store_true",
      help="Run stitching in the preview mode")
  parser.add_argument("--work_megapix", dest="work_megapix", default=0.6, type=float,
      help="Resolution for image registration step. The default is %(default)s Mpx")
  parser.add_argument("-f", "--features", dest="features", default="orb", type=str,
      help="Type of features used for images matching. The default is %(default)s")

  args = parser.parse_args()

  print("Args")
  print("  img_names: {}".format(args.img_names))
  print("  preview: {}".format(args.preview))
  print("  work_megapix: {}".format(args.work_megapix))
  print("  features: {}".format(args.features))

  return args


def _main():
  args = _parse_args()
  img_names = args.img_names

  if not img_names:
    sys.exit("python features_finding.py boat*.jpg")

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

  work_megapix = args.work_megapix
  is_work_scale_set = False

  for name in img_names:
    full_img = cv.imread(cv.samples.findFile(name))
    if full_img is None:
      sys.exit(f"Cannot read image {name}")

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

    features = cv.detail.computeImageFeatures2(finder, img)
    keypoints = features.getKeypoints()
    # keypoints, _ = finder.detectAndCompute(img, None)

    print(f"{name} {img.shape[1]}x{img.shape[0]}, keypoints: {len(keypoints)}, "
          f"cost: {time.time()-t:.2f} s")

    if args.preview:
      cv.drawKeypoints(img, keypoints, img)

      cv.imshow(f"{name} {img.shape[1]}x{img.shape[0]}", img)
      cv.waitKey(0)

  cv.destroyAllWindows()


if __name__ == "__main__":
  _main()
