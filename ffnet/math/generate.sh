#!/bin/bash

PATH="$PATH:./bin"
LLVM_FLAGS="-mllvm -inline-threshold=1000"
#CLANG_OPTS="-mno-red-zone -no-integrated-as -mstackrealign -fno-asynchronous-unwind-tables -fno-exceptions -fno-rtti -ffast-math -O3"
#CLANG_OPTS="-mfma -no-builtin -ffreestanding -mno-red-zone -no-integrated-as -mstackrealign -fno-asynchronous-unwind-tables -fno-exceptions -fno-rtti -ffast-math -O3"
CLANG_OPTS="-mfma -mstackrealign -fno-asynchronous-unwind-tables -fno-exceptions -fno-rtti -ffast-math -O3"


function build_avx2_amd64 {
    SRC="avx2_amd64.cpp"
    ASM="avx2_amd64.s"
    clang-15 -S -mavx2 -masm=intel $CLANG_OPTS $LLVM_FLAGS -o $ASM $SRC
    c2goasm -a -f $ASM ../$ASM
   # rm $ASM
}

function build_neon_arm64 {
    SRC="neon_arm64.cpp"
    ASM="neon_arm64.s"
    clang-14 -S -arch arm64 $CLANG_OPTS $LLVM_FLAGS -o $ASM $SRC
    c2goasm -a -f $ASM ../$ASM
    rm $ASM
}

build_avx2_amd64
#build_avx512_amd64
#build_neon_arm64
