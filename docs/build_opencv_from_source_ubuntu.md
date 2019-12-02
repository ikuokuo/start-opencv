# Build OpenCV from source

> Ubuntu

## Get Deps

Python,

```bash
# https://www.anaconda.com/distribution/
bash Anaconda3-2019.10-Linux-x86_64.sh
```

<!--
/home/john/anaconda3/bin/conda init
conda config --set auto_activate_base true
-->

CUDA,

```bash
cat <<EOF >>~/.bashrc
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
EOF
```

## Get OpenCV

```bash
git clone -b 4.1.2 https://github.com/opencv/opencv.git
git clone -b 4.1.2 https://github.com/opencv/opencv_contrib.git
```

## Build OpenCV

<!--
Download,
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/32e315a5b106a7b89dbed51c28f8120a48b368b4/ippicv/ippicv_2019_lnx_intel64_general_20180723.tgz
To,
  $HOME/Downloads/ippicv/
export OPENCV_IPPICV_URL=file://$HOME/Downloads
-->

```bash
export SYS_PREFIX=`conda info -s | grep -Po "sys.prefix:\s*\K[/\w]*"`
export OPENCV_CONTRIB=$HOME/Workspace/opencv_contrib

mkdir _build; cd _build

cmake -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=$HOME/opencv-4.1.2 \
\
-DOPENCV_ENABLE_NONFREE=ON \
-DOPENCV_EXTRA_MODULES_PATH=$OPENCV_CONTRIB/modules \
\
-DPYTHON_EXECUTABLE=$SYS_PREFIX/bin/python3.7 \
-DPYTHON3_EXECUTABLE=$SYS_PREFIX/bin/python3.7 \
-DPYTHON3_LIBRARY=$SYS_PREFIX/lib/libpython3.7m.so \
-DPYTHON3_INCLUDE_DIR=$SYS_PREFIX/include/python3.7m \
-DPYTHON3_NUMPY_INCLUDE_DIRS=$SYS_PREFIX/lib/python3.7/site-packages/numpy/core/include \
-DBUILD_opencv_python2=OFF \
-DBUILD_opencv_python3=ON \
\
-DWITH_CUDA=ON \
\
-DBUILD_DOCS=OFF \
-DBUILD_EXAMPLES=OFF \
-DBUILD_TESTS=OFF \
..

make -j2
make install
```

### References

* [Tensorflow crashes on build on Ubuntu 16.04 when building for skylake (avx512)](https://github.com/tensorflow/tensorflow/issues/10220)

GCC 5.4.0,

```bash
bash install_gcc54.sh

export LD_LIBRARY_PATH=/opt/gnu/mpc-1.1.0/lib:/opt/gnu/mpfr-4.0.2/lib:/opt/gnu/gmp-6.1.2/lib:$LD_LIBRARY_PATH

# rm _build/CMakeCache.txt
cmake \
-DCMAKE_C_COMPILER=/opt/gnu/gcc-5.4.0/bin/gcc \
-DCMAKE_CXX_COMPILER=/opt/gnu/gcc-5.4.0/bin/g++ \
..
```

## Start OpenCV

```bash
export PYTHONPATH=$HOME/opencv-4.1.2/lib/python3.7/site-packages:$PYTHONPATH
python - <<EOF
import cv2
print(cv2.__version__)
EOF
```
