#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import os
import sys
import time

import cv2 as cv
import numpy as np


def _parse_args():
  import argparse
  parser = argparse.ArgumentParser(
      usage="python src/python/stitching/images_matching.py data/stitching/boat*.jpg -p")

  parser.add_argument("img_names", nargs="+", type=str, help="files to stitch")
  parser.add_argument("-p", "--preview", dest="preview", action="store_true",
      help="Run stitching in the preview mode")
  parser.add_argument("--work_megapix", dest="work_megapix", default=0.6, type=float,
      help="Resolution for image registration step. The default is %(default)s Mpx")
  parser.add_argument("-f", "--features", dest="features", default="orb", type=str,
      help="Type of features used for images matching. The default is %(default)s")

  parser.add_argument("--try_cuda", dest="try_cuda", action="store_true",
      help="Try to use CUDA")
  parser.add_argument("--matcher", dest="matcher", default="homography", type=str,
      help="Matcher used for pairwise image matching")
  parser.add_argument("--match_conf", dest="match_conf", default=0.3, type=float,
      help="Confidence for feature matching step. The default is 0.65 for surf and 0.3 for orb")
  parser.add_argument("--rangewidth", dest="rangewidth", default=-1, type=int,
      help="Use range_width to limit number of images to match with")
  parser.add_argument("--conf_thresh", dest="conf_thresh", default=1.0, type=float,
      help="Threshold for two images are from the same panorama confidence.The default is 1.0")
  parser.add_argument("--save_graph", dest="save_graph", default=None, type=str,
      help="Save matches graph represented in DOT language to <file_name> file")

  args = parser.parse_args()

  print("Args")
  print("  img_names: {}".format(args.img_names))
  print("  preview: {}".format(args.preview))
  print("  work_megapix: {}".format(args.work_megapix))
  print("  features: {}".format(args.features))

  print("  try_cuda: {}".format(args.try_cuda))
  print("  matcher: {}".format(args.matcher))
  print("  match_conf: {}".format(args.match_conf))
  print("  rangewidth: {}".format(args.rangewidth))
  print("  conf_thresh: {}".format(args.conf_thresh))
  print("  save_graph: {}".format(args.save_graph))

  return args


def _main():
  args = _parse_args()
  img_names = args.img_names

  if not img_names:
    sys.exit("python images_matching.py boat*.jpg")

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

    if args.preview:
      img = cv.drawKeypoints(img, keypoints, None)

      cv.imshow(f"{name} {img.shape[1]}x{img.shape[0]}", img)
      cv.waitKey(0)

  cv.destroyAllWindows()

  # matching

  try_cuda = args.try_cuda
  matcher_type = args.matcher
  match_conf = args.match_conf
  range_width = args.rangewidth
  conf_thresh = args.conf_thresh
  if args.save_graph is None:
    save_graph = False
  else:
    save_graph = True
    save_graph_to = args.save_graph

  if matcher_type == "affine":
    matcher = cv.detail_AffineBestOf2NearestMatcher(False, try_cuda, match_conf)
  elif range_width == -1:
    matcher = cv.detail.BestOf2NearestMatcher_create(try_cuda, match_conf)
  else:
    matcher = cv.detail.BestOf2NearestRangeMatcher_create(range_width, try_cuda, match_conf)

  matches = matcher.apply2(features)
  matcher.collectGarbage()

  num_images = len(img_names)
  print(f"\nMatches size: {num_images}")
  for i in range(num_images):
    print(f"  {i} {os.path.basename(img_names[i])} >")
    for j in range(num_images):
      info = matches[i*num_images + j]
      print(f"    {info.dst_img_idx}, conf: {info.confidence}")

  if save_graph:
    # Image Stitching details with OpenCV
    #  https://stackoverflow.com/questions/26364594/image-stitching-details-with-opencv
    f = open(save_graph_to, "w")
    f.write(cv.detail.matchesGraphAsString(img_names, matches, conf_thresh))
    f.close()

  indices = cv.detail.leaveBiggestComponent(features, matches, 0.3)
  print(f"Matches indices: {[i[0] for i in indices]}")

  img_subset = []
  img_names_subset = []
  full_img_sizes_subset = []

  for i in range(len(indices)):
    indice = indices[i, 0]
    img_subset.append(images[indice])
    img_names_subset.append(img_names[indice])
    full_img_sizes_subset.append(full_img_sizes[indice])

  images = img_subset
  img_names = img_names_subset
  full_img_sizes = full_img_sizes_subset

  num_images = len(img_names)
  if num_images < 2:
    sys.exit("Need more images")

  print("Matches subsets")
  print(f"  img_names: {img_names}")
  print(f"  full_img_sizes: {full_img_sizes}")


if __name__ == "__main__":
  _main()
