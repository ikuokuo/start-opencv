# Build OpenCV from source

> macOS

## Get Deps

Python,

```bash
brew install pyenv
# brew upgrade pyenv

pyenv install anaconda3-2020.07
pyenv global anaconda3-2020.07
```

<!--
env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.9.0
pyenv global 3.9.0
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
git clone -b 4.5.0 --depth 1 https://github.com/opencv/opencv.git
git clone -b 4.5.0 --depth 1 https://github.com/opencv/opencv_contrib.git
```

## Build OpenCV

```bash
export PYENV_PREFIX=`pyenv prefix`
export OPENCV_CONTRIB=$HOME/Workspace/Star/opencv_contrib

mkdir _build; cd _build

cmake -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=$HOME/opencv-4.5.0 \
\
-DOPENCV_ENABLE_NONFREE=ON \
-DOPENCV_EXTRA_MODULES_PATH=$OPENCV_CONTRIB/modules \
\
-DPYTHON_EXECUTABLE=$PYENV_PREFIX/bin/python3.8 \
-DPYTHON3_EXECUTABLE=$PYENV_PREFIX/bin/python3.8 \
-DPYTHON3_LIBRARY=$PYENV_PREFIX/lib/libpython3.8.dylib \
-DPYTHON3_INCLUDE_DIR=$PYENV_PREFIX/include/python3.8 \
-DPYTHON3_NUMPY_INCLUDE_DIRS=$PYENV_PREFIX/lib/python3.8/site-packages/numpy/core/include \
-DBUILD_opencv_python2=OFF \
-DBUILD_opencv_python3=ON \
\
-DBUILD_DOCS=OFF \
-DBUILD_EXAMPLES=OFF \
-DBUILD_TESTS=OFF \
..

make -j`nproc`
make install
```

## Start OpenCV

```bash
export PYTHONPATH=$HOME/opencv-4.5.0/lib/python3.8/site-packages:$PYTHONPATH
python - <<EOF
import cv2
print(cv2.__version__)
EOF
```
