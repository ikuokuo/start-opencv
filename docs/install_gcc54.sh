#!/usr/bin/env bash

BASE_DIR=$(cd "$(dirname "$0")" && pwd)

OUT_DIR=/opt/gnu

SUDO=sudo

GMP=gmp-6.1.2
MPFR=mpfr-4.0.2
MPC=mpc-1.1.0
GCC=gcc-5.4.0

cd $BASE_DIR
if [ ! -d $GMP ]; then
  wget https://mirrors.ustc.edu.cn/gnu/gmp/$GMP.tar.bz2
  tar xjf $GMP.tar.bz2
fi
cd $GMP
./configure --prefix=$OUT_DIR/$GMP --enable-cxx CPPFLAGS=-fexceptions
make -j4; $SUDO make install

cd $BASE_DIR
if [ ! -d $MPFR ]; then
  wget https://mirrors.ustc.edu.cn/gnu/mpfr/$MPFR.tar.bz2
  tar xjf $MPFR.tar.bz2
fi
cd $MPFR
./configure --prefix=$OUT_DIR/$MPFR --with-gmp=$OUT_DIR/$GMP
make -j4; $SUDO make install

cd $BASE_DIR
if [ ! -d $MPC ]; then
  wget https://mirrors.ustc.edu.cn/gnu/mpc/$MPC.tar.gz
  tar xzf $MPC.tar.gz
fi
cd $MPC
./configure --prefix=$OUT_DIR/$MPC --with-gmp=$OUT_DIR/$GMP --with-mpfr=$OUT_DIR/$MPFR
make -j4; $SUDO make install

export LD_LIBRARY_PATH=$OUT_DIR/$MPC/lib:$OUT_DIR/$MPFR/lib:$OUT_DIR/$GMP/lib:$LD_LIBRARY_PATH

cd $BASE_DIR
if [ ! -d $GCC ]; then
  wget https://mirrors.ustc.edu.cn/gnu/gcc/$GCC/$GCC.tar.gz
  tar xzf $GCC.tar.gz
fi
cd $GCC
./configure \
--prefix=$OUT_DIR/$GCC \
--with-gmp=$OUT_DIR/$GMP \
--with-mpfr=$OUT_DIR/$MPFR \
--with-mpc=$OUT_DIR/$MPC \
--enable-languages=c,c++ \
--disable-multilib
make -j4; $SUDO make install
