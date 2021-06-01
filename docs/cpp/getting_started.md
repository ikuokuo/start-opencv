# Getting Started

## Get OpenCV

### Ubuntu

[Build OpenCV from source](../build_opencv_from_source_ubuntu.md),

```bash
export OpenCV_DIR=$HOME/opencv-4/lib/cmake
```

### macOS

[Build OpenCV from source](../build_opencv_from_source_macos.md),

```bash
export OpenCV_DIR=$HOME/opencv-4/lib/cmake
```

Or install OpenCV with `brew`,

```bash
brew install opencv
export OpenCV_DIR=`brew --prefix opencv`/lib/cmake
```

## Start OpenCV

```bash
make
./_output/bin/camera/camera
```
