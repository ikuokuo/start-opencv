# Build OpenCV from source

> macOS

## Get Deps

Python,

```bash
brew install pyenv

pyenv install anaconda3-2019.10
pyenv global anaconda3-2019.10
```

<!--
env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.7.0
pyenv global 3.7.0
pip install numpy
-->

<!--
CUDA,

```bash
brew tap caskroom/drivers
brew cask install nvidia-cuda

cat <<EOF >>~/.bash_profile
export CUDA_HOME=/Developer/NVIDIA/CUDA-10.1
export PATH=\$CUDA_HOME/bin:\$PATH
export DYLD_LIBRARY_PATH=\$CUDA_HOME/lib:\$DYLD_LIBRARY_PATH
EOF
```
-->

## Get OpenCV

```bash
git clone -b 4.3.0 https://github.com/opencv/opencv.git
git clone -b 4.3.0 https://github.com/opencv/opencv_contrib.git
```

## Build OpenCV

<!--
Download,
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/a56b6ac6f030c312b2dce17430eef13aed9af274/ippicv/ippicv_2020_mac_intel64_20191018_general.tgz
To,
  $HOME/Downloads/ippicv/
export OPENCV_IPPICV_URL=file://$HOME/Downloads
-->

```bash
export PYENV_PREFIX=`pyenv prefix`
export OPENCV_CONTRIB=$HOME/Workspace/Star/opencv_contrib

mkdir _build; cd _build

cmake -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=$HOME/opencv-4.3.0 \
\
-DOPENCV_ENABLE_NONFREE=ON \
-DOPENCV_EXTRA_MODULES_PATH=$OPENCV_CONTRIB/modules \
\
-DPYTHON_EXECUTABLE=$PYENV_PREFIX/bin/python3.7 \
-DPYTHON3_EXECUTABLE=$PYENV_PREFIX/bin/python3.7 \
-DPYTHON3_LIBRARY=$PYENV_PREFIX/lib/libpython3.7m.dylib \
-DPYTHON3_INCLUDE_DIR=$PYENV_PREFIX/include/python3.7m \
-DPYTHON3_NUMPY_INCLUDE_DIRS=$PYENV_PREFIX/lib/python3.7/site-packages/numpy/core/include \
-DBUILD_opencv_python2=OFF \
-DBUILD_opencv_python3=ON \
\
-DBUILD_DOCS=OFF \
-DBUILD_EXAMPLES=OFF \
-DBUILD_TESTS=OFF \
..

make -j2
make install
```

## Start OpenCV

```bash
export PYTHONPATH=$HOME/opencv-4.3.0/lib/python3.7/site-packages:$PYTHONPATH
python - <<EOF
import cv2
print(cv2.__version__)
EOF
```
