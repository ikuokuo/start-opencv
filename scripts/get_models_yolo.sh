#!/usr/bin/env bash
# e.g.
#   bash scripts/get_models_yolo.sh

BASE_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(realpath "$BASE_DIR/..")

out_dir=$ROOT_DIR/data/detection
if [ ! -d $out_dir ]; then
  mkdir -p $out_dir
fi

cd  $out_dir/
[ -e yolov3.weights ] || wget https://pjreddie.com/media/files/yolov3.weights
[ -e yolov3.cfg ] || wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true -O ./yolov3.cfg
[ -e coco.names ] || wget https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true -O ./coco.names
