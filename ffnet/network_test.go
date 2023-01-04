package ffnet

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/kelindar/evolve"
	"github.com/stretchr/testify/assert"
)

/*
cpu: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz
BenchmarkPredict/10x2x1-8         	33971622	        35.35 ns/op	       0 B/op	       0 allocs/op
BenchmarkPredict/10x10x1-8        	27273470	        44.41 ns/op	       0 B/op	       0 allocs/op
BenchmarkPredict/10x100x1-8       	 6984877	       173.8 ns/op	       0 B/op	       0 allocs/op
BenchmarkPredict/10x1000x1-8      	 1000000	      1329 ns/op	       0 B/op	       0 allocs/op
BenchmarkPredict/10x10000x1-8     	   88837	     13396 ns/op	       0 B/op	       0 allocs/op
*/
func BenchmarkPredict(b *testing.B) {
	for _, size := range []int{2, 10} {
		b.Run(fmt.Sprintf("10x%dx1", size), func(b *testing.B) {
			nn := NewFeedForward([]int{10, size, size, 1})
			in := make([]float32, 10)
			out := make([]float32, 1)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				nn.Predict(in, out)
			}
		})
	}

	for size, count := range sizes {
		fmt.Printf("size=%v, count=%v\n", size, count)
	}
}

/*
cpu: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz
BenchmarkEvolve-8   	   25706	     46466 ns/op	       0 B/op	       0 allocs/op
*/
func BenchmarkEvolve(b *testing.B) {
	pop := evolve.New(256, func(*FeedForward) float32 { return 0 }, func() *FeedForward {
		return NewFeedForward([]int{3, 2, 1})
	})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pop.Evolve()
	}
}

/*
cpu: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz
BenchmarkDeep-8   	      74	  16630714 ns/op	      13 B/op	       0 allocs/op
*/
func BenchmarkDeep(b *testing.B) {
	const layers = 512
	shape := make([]int, 0, layers)
	shape = append(shape, 10)
	for i := 0; i < layers-2; i++ {
		shape = append(shape, 128)
	}
	shape = append(shape, 1)

	nn := NewFeedForward(shape)
	in := make([]float32, 10)
	out := make([]float32, 1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nn.Predict(in, out)
	}
}

func TestXOR(t *testing.T) {
	nn := newXOR()
	tests := []struct {
		input  []float32
		output float32
	}{
		{input: []float32{1, 0, 0}, output: 0},
		{input: []float32{1, 0, 1}, output: 1},
		{input: []float32{1, 1, 0}, output: 1},
		{input: []float32{1, 1, 1}, output: 0},
	}
	for _, tc := range tests {
		const delta = 0.05
		out := nn.Predict(tc.input, nil)[0]
		assert.InDelta(t, tc.output, out, delta)
	}
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

	matmul(o, x, y, 2, 2, 2, 2)
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

func newXOR() *FeedForward {
	return NewFeedForward([]int{3, 2, 1},
		[]float32{-0.21966185, 0.77679425, -0.9666261, -0.16248938, -1.5812368, 1.7551233},
		[]float32{1.814858, 0.6414089},
	)
}

// axpyRef function, this doesn't use any SIMD as it seems like this version
// is actually faster than blas32 one from gonum
func axpyRef(x, y []float32, alpha float32) {
	_ = y[len(x)-1] // remove bounds checks
	for i, v := range x {
		y[i] += alpha * v
	}
}
