#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import argparse
import glob
import multiprocessing as mp
import os
import time
import tqdm

import cv2 as cv
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

from detectron2_predictor import VisualizationDemo


def _setup_cfg(args):
  # load config from file and command-line arguments
  cfg = get_cfg()
  # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
  # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
  # add_panoptic_deeplab_config(cfg)
  cfg.merge_from_file(args.config_file)
  cfg.merge_from_list(args.opts)
  # Set score_threshold for builtin models
  cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
  cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
  cfg.freeze()
  return cfg


def _parse_args():
  parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
  parser.add_argument(
    "--config-file",
    default="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    metavar="FILE",
    help="path to config file",
  )
  parser.add_argument(
    "--input",
    nargs="+",
    required=True,
    help="A list of space separated input images; "
    "or a single glob pattern such as 'directory/*.jpg'",
  )
  parser.add_argument(
    "--output",
    help="A file or directory to save output visualizations. "
    "If not given, will show output in an OpenCV window.",
  )
  parser.add_argument(
    "--confidence-threshold",
    type=float,
    default=0.5,
    help="Minimum score for instance predictions to be shown",
  )
  parser.add_argument(
    "--opts",
    default=[],
    nargs=argparse.REMAINDER,
    help="Modify config options using the command-line 'KEY VALUE' pairs",
  )

  args = parser.parse_args()

  print("Args")
  print(f"  config_file: {args.config_file}")
  print(f"  input: {args.input}")
  print(f"  output: {args.output}")
  print(f"  confidence_threshold: {args.confidence_threshold}")
  print(f"  opts: {args.opts}")

  return args


def _save_thresholding(class_names, predictions, filepath):
  if "instances" not in predictions:
    return
  instances = predictions["instances"].to(torch.device("cpu"))
  assert instances.has("pred_classes") and instances.has("pred_masks")

  classes = instances.pred_classes.tolist()
  masks = np.asarray(instances.pred_masks)

  print("Save thresholding:")
  for cls, mask in zip(classes, masks):
    savepath = f"{os.path.splitext(filepath)[0]}_thres_{class_names[cls]}.png"
    print(f"  {savepath}")
    img = np.zeros(instances.image_size)
    img[mask] = 255
    cv.imwrite(savepath, img)


def _main():
  mp.set_start_method("spawn", force=True)
  args = _parse_args()

  cfg = _setup_cfg(args)

  demo = VisualizationDemo(cfg)

  class_names = demo.metadata.thing_classes

  if len(args.input) == 1:
    args.input = glob.glob(os.path.expanduser(args.input[0]))
    assert args.input, "The input path(s) was not found"
  for path in tqdm.tqdm(args.input, disable=not args.output):
    # use PIL, to be consistent with evaluation
    img = read_image(path, format="BGR")
    start_time = time.time()
    predictions, visualized_output = demo.run_on_image(img)
    print(
        "{}: {} in {:.2f}s".format(
            path,
            "detected {} instances".format(len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            time.time() - start_time,
        )
    )

    if args.output:
      if os.path.isdir(args.output):
        assert os.path.isdir(args.output), args.output
        out_filename = os.path.join(args.output, os.path.basename(path))
      else:
        assert len(args.input) == 1, "Please specify a directory with args.output"
        out_filename = args.output
      visualized_output.save(out_filename)

      _save_thresholding(class_names, predictions, out_filename)
    else:
      win_name = "output"
      cv.namedWindow(win_name, cv.WINDOW_NORMAL)
      cv.imshow(win_name, visualized_output.get_image()[:, :, ::-1])
      key = cv.waitKey(0) & 0xFF
      if key == 27 or key == ord('q'):
        break


if __name__ == "__main__":
  _main()


# Detectron2 开始: https://yyixx.com/docs/algo/detectron2

# export DETECTRON2_DIR=detectron2
# export DETECTRON2_MODELS_DIR=models
# mkdir -p _output
# python detectron2_seg_threshold.py --config-file $DETECTRON2_DIR/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input ../data/processing/dog_catch_ball.jpg --output _output --confidence-threshold 0.6 --opts MODEL.WEIGHTS $DETECTRON2_MODELS_DIR/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
