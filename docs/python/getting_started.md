# Getting Started

## Get OpenCV

### Ubuntu

[Build OpenCV from source](../build_opencv_from_source_ubuntu.md),

```bash
export PYTHONPATH=$HOME/opencv-4.1.2/lib/python3.7/site-packages:`pwd`/src/python:$PYTHONPATH
```

### macOS

[Build OpenCV from source](../build_opencv_from_source_macos.md),

```bash
export PYTHONPATH=$HOME/opencv-4.3.0/lib/python3.7/site-packages:`pwd`/src/python:$PYTHONPATH
```

Or install OpenCV with `brew`,

```bash
brew install opencv
export PYTHONPATH=`brew --prefix opencv`/lib/python3.7/site-packages:`pwd`/src/python:$PYTHONPATH
```

## Start OpenCV

```bash
python src/python/common/camera.py
```
