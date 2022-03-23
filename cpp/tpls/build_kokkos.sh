#!/usr/bin/env bash

set -e

function bk(){
    local btype=$1

    [[ ! -d ${KOKKOS_BUILD_DIR} ]] && mkdir -p ${KOKKOS_BUILD_DIR}
    cd ${KOKKOS_BUILD_DIR} #&& rm -rf CMakeCache* core/*

	cmake -DCMAKE_CXX_COMPILER=${CXX} \
	      -DCMAKE_BUILD_TYPE="${btype}" \
	      -DCMAKE_INSTALL_PREFIX=${KOKKOS_PFX} \
	      -DKokkos_ENABLE_TESTS=Off \
	      -DKokkos_ENABLE_EXAMPLES=Off \
	      -DKokkos_ENABLE_OPENMP=On \
	      -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=On \
	      ${KOKKOS_SRC}

    make -j10
    make install
    cd ..
}

function bkk(){
    echo "Kernels using the KokkosPFX= ${KOKKOSPFX}"
    local btype=$1

    [[ ! -d ${KOKKOS_K_BUILD_DIR} ]] && mkdir -p ${KOKKOS_K_BUILD_DIR}
    cd ${KOKKOS_K_BUILD_DIR} && rm -rf CMakeCache* src/*

    cmake \
	-DCMAKE_VERBOSE_MAKEFILE=On \
	-DCMAKE_CXX_COMPILER=${CXX} \
	-DCMAKE_BUILD_TYPE="${btype}" \
	-DCMAKE_INSTALL_PREFIX=${KOKKOS_K_PFX} \
	-DKokkos_ROOT=${KOKKOS_PFX} \
	\
	-DKokkosKernels_ENABLE_TPL_LAPACK=On \
	-DKokkosKernels_ENABLE_TPL_BLAS=On \
	\
	-DKokkosKernels_INST_DOUBLE=On \
	-DKokkosKernels_INST_LAYOUTRIGHT=On \
	-DKokkosKernels_INST_LAYOUTLEFT=On \
	\
	-DKokkosKernels_ENABLE_TESTS=On \
	${KOKKOS_K_SRC}

    make -j10
    make install
    cd ..
}

MYPWD=`pwd`

KOKKOS_SRC=${MYPWD}/kokkos-3.5.00
KOKKOS_BUILD_DIR=${MYPWD}/build_kokkos
KOKKOS_PFX=${MYPWD}/install_kokkos

KOKKOS_K_SRC=${MYPWD}/kokkos-kernels-3.5.00
KOKKOS_K_BUILD_DIR=${MYPWD}/build_kokkos_ker
KOKKOS_K_PFX=${MYPWD}/install_kokkos_ker

bk  release
bkk release

cd ${MYPWD}


# -DKokkosKernels_INST_ORDINAL_INT=Off \
# -DKokkosKernels_INST_ORDINAL_INT64_T=On \
# -DKokkosKernels_INST_OFFSET_INT=Off \
# -DKokkosKernels_INST_OFFSET_SIZE_T=On \
