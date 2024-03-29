# Build OpenCV from source

> Ubuntu

## Get Deps

Python,

```bash
# https://www.anaconda.com/distribution/
bash Anaconda3-2021.11-Linux-x86_64.sh
```

<!--
/home/john/anaconda3/bin/conda init
conda config --set auto_activate_base true
-->

CUDA,

* [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
* [cuDNN Download](https://developer.nvidia.com/rdp/cudnn-download)

```bash
cat <<EOF >>~/.bashrc
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
EOF
```

## Get OpenCV

```bash
git clone -b 4.5.5 --depth 1 https://github.com/opencv/opencv.git
git clone -b 4.5.5 --depth 1 https://github.com/opencv/opencv_contrib.git
```

## Build OpenCV

```bash
# GUI, GTK+:
#  sudo apt install -y libgtk-3-dev
# Video I/O, FFMPEG:
#  sudo apt install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavresample-dev

conda deactivate

export CONDA_HOME=`conda info -s | grep -Po "sys.prefix:\s*\K[/\w]*"`
export OPENCV_CONTRIB=$HOME/Codes/Star/opencv_contrib

cd $HOME/Codes/Star/opencv/
mkdir _build; cd _build

cmake -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=$HOME/opencv-4.5.5 \
-DOPENCV_EXTRA_MODULES_PATH=$OPENCV_CONTRIB/modules \
\
-DPYTHON_EXECUTABLE=$CONDA_HOME/bin/python3.9 \
-DPYTHON3_EXECUTABLE=$CONDA_HOME/bin/python3.9 \
-DPYTHON3_LIBRARY=$CONDA_HOME/lib/libpython3.9.so \
-DPYTHON3_INCLUDE_DIR=$CONDA_HOME/include/python3.9 \
-DPYTHON3_NUMPY_INCLUDE_DIRS=$CONDA_HOME/lib/python3.9/site-packages/numpy/core/include \
-DBUILD_opencv_python2=OFF \
-DBUILD_opencv_python3=ON \
\
-DWITH_CUDA=ON \
\
-DBUILD_DOCS=OFF \
-DBUILD_EXAMPLES=OFF \
-DBUILD_TESTS=OFF \
..

make -j$(nproc)
make install

cd $HOME
ln -sf opencv-4.5.5 opencv-4

conda activate
```

<!--
/usr/local/cuda/extras/demo_suite/deviceQuery

-DCUDA_ARCH_BIN="7.5" \
-DCUDA_ARCH_PTX="" \
-->

## Start OpenCV

```bash
export PYTHONPATH=$HOME/opencv-4/lib/python3.9/site-packages:$PYTHONPATH
python - <<EOF
import cv2
print(cv2.__version__)
EOF
```

### ImportError: libgio-2.0.so.0

```bash
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/john/opencv-4/lib/python3.9/site-packages/cv2/__init__.py", line 180, in <module>
    bootstrap()
  File "/home/john/opencv-4/lib/python3.9/site-packages/cv2/__init__.py", line 152, in bootstrap
    native_module = importlib.import_module("cv2")
  File "/home/john/anaconda3/lib/python3.9/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
ImportError: /home/john/anaconda3/bin/../lib/libgio-2.0.so.0: undefined symbol: g_uri_join
```

Solution:

```bash
cd $HOME/anaconda3/lib/
mv libgio-2.0.so.0 libgio-2.0.so.0.bak
ln -s /usr/lib/x86_64-linux-gnu/libgio-2.0.so.0 libgio-2.0.so.0
```
