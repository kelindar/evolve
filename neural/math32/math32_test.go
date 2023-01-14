// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package math32

import (
	"fmt"
	"math"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
)

/*
cpu: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz
BenchmarkAXPY/std-8         	168842340	         7.228 ns/op	       0 B/op	       0 allocs/op
BenchmarkAXPY/asm-8         	189254059	         6.233 ns/op	       0 B/op	       0 allocs/op
*/
func BenchmarkAXPY(b *testing.B) {
	x := []float32{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}
	y := []float32{3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}

	b.Run("std", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			axpyRef(x, y, 3)
		}
	})

	b.Run("asm", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_f32_axpy(
				unsafe.Pointer(&x[0]),
				unsafe.Pointer(&y[0]),
				4, 3.0,
			)
		}
	})
}

/*
cpu: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz
BenchmarkMatmul/4x4-std-8         	17968401	        68.36 ns/op	       0 B/op	       0 allocs/op
BenchmarkMatmul/4x4-asm-8         	30570522	        42.56 ns/op	       0 B/op	       0 allocs/op
BenchmarkMatmul/8x8-std-8         	 3465379	       349.4 ns/op	       0 B/op	       0 allocs/op
BenchmarkMatmul/8x8-asm-8         	13685428	        84.99 ns/op	       0 B/op	       0 allocs/op
BenchmarkMatmul/16x16-std-8       	  590862	      2279 ns/op	       0 B/op	       0 allocs/op
BenchmarkMatmul/16x16-asm-8       	 3117709	       389.1 ns/op	       0 B/op	       0 allocs/op
BenchmarkMatmul/32x32-std-8       	   73831	     15662 ns/op	       0 B/op	       0 allocs/op
BenchmarkMatmul/32x32-asm-8       	  594648	      2435 ns/op	       0 B/op	       0 allocs/op
BenchmarkMatmul/64x64-std-8       	   10000	    117896 ns/op	       0 B/op	       0 allocs/op
BenchmarkMatmul/64x64-asm-8       	   90950	     14022 ns/op	       0 B/op	       0 allocs/op
*/
func BenchmarkMatmul(b *testing.B) {
	for _, size := range []int{4, 8, 16, 32, 64} {
		m := newTestMatrix(size, size)
		n := newTestMatrix(size, size)
		o := newTestMatrix(size, size)

		b.Run(fmt.Sprintf("%dx%d-std", size, size), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_matmul(o.Data, m.Data, n.Data, m.Rows, m.Cols, n.Rows, n.Cols)
			}
		})

		b.Run(fmt.Sprintf("%dx%d-asm", size, size), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_f32_matmul(
					unsafe.Pointer(&o.Data[0]), unsafe.Pointer(&m.Data[0]), unsafe.Pointer(&n.Data[0]),
					uint64(m.Rows), uint64(m.Cols), uint64(n.Rows), uint64(n.Cols),
				)
			}
		})
	}
}

func TestApproxSwish(t *testing.T) {
	mae := testApproxSwish(-10, 10)
	assert.InDelta(t, 0, mae, 0.10, "unexpected MAE: %.5f", mae)
}

func testApproxSwish(start, end float32) float64 {
	err, n := 0.0, 0
	for x := start; x <= end; x += 0.05 {
		ratio := referenceSwish(x) / swish(x)
		if referenceSwish(x) == 0 {
			continue // skip
		}

		n++
		err += math.Abs(float64(ratio) - 1)
	}

	// Compute mean absolute error
	return err / float64(n)
}

func referenceSwish(x float32) float32 {
	return x / (1.0 + float32(math.Exp(float64(-x))))
}

func TestAsmMatmul(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	y := []float32{5, 6, 7, 8}
	o := make([]float32, 4)

	_f32_matmul(
		unsafe.Pointer(&o[0]), unsafe.Pointer(&x[0]), unsafe.Pointer(&y[0]),
		2, 2, 2, 2)

	assert.Equal(t, []float32{19, 22, 43, 50}, o)
}

func TestGenericMatmul(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	y := []float32{5, 6, 7, 8}
	o := make([]float32, 4)

	_matmul(o, x, y, 2, 2, 2, 2)
	assert.Equal(t, []float32{19, 22, 43, 50}, o)
}

func TestAXPY(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	y := []float32{1, 1, 1, 1}

	_f32_axpy(
		unsafe.Pointer(&x[0]),
		unsafe.Pointer(&y[0]),
		4, 2,
	)

	_ = x[0]
	_ = y[0]
	assert.Equal(t, []float32{3, 5, 7, 9}, y)
}

// axpyRef function, this doesn't use any SIMD as it seems like this version
// is actually faster than blas32 one from gonum
func axpyRef(x, y []float32, alpha float32) {
	_ = y[len(x)-1] // remove bounds checks
	for i, v := range x {
		y[i] += alpha * v
	}
}

// newTestMatrix creates a new matrix
func newTestMatrix(r, c int) *Matrix {
	mx := NewDense(r, c, nil)
	for i := 0; i < len(mx.Data); i++ {
		mx.Data[i] = 2
	}
	return &mx
}
