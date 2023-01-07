#!/bin/bash

# relevant clang flags:
# -mavx2: enables AVX2 instructions for the compiled code
# -masm=intel: specifies to use the Intel syntax for assembly language
# -mllvm -inline-threshold=1000: tells the compiler to use the LLVM inline threshold of 1000 for optimization
# -mfma: enables FMA instructions for the compiled code
# -mstackrealign: tells the compiler to realign the stack before function calls
# -mno-red-zone: disables the use of the red zone in the compiled code
# -fno-asynchronous-unwind-tables: tells the compiler not to generate asynchronous unwind tables
# -fno-exceptions: disables the use of exceptions in the compiled code
# -fno-rtti: disables the use of RTTI (run-time type information) in the compiled code
# -ffast-math: enables "unsafe" floating point optimizations
# -O3: enables level 3 optimization for the compiled code
PATH="$PATH:./bin"

function build_avx2_amd64 {
    SRC="avx2_amd64.cpp"
    ASM="avx2_amd64.s"
    clang-15 -S -o $ASM $SRC \
        -mavx2 \ 
        -masm=intel \
        -mllvm -inline-threshold=1000
        -mfma \
        -mstackrealign \
        -mno-red-zone \
        -fno-asynchronous-unwind-tables \
        -fno-exceptions \
        -fno-rtti \
        -ffast-math \
        -O3
    c2goasm -a -f $ASM ../$ASM
    rm $ASM
}

build_avx2_amd64