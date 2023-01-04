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
BenchmarkPredict/10x2x1-8         	21970818	        54.49 ns/op	       0 B/op	       0 allocs/op
BenchmarkPredict/10x10x1-8        	13367792	        86.16 ns/op	       0 B/op	       0 allocs/op
BenchmarkPredict/10x100x1-8       	 2998368	       399.0 ns/op	       0 B/op	       0 allocs/op
BenchmarkPredict/10x1000x1-8      	  319666	      3729 ns/op	       0 B/op	       0 allocs/op
BenchmarkPredict/10x10000x1-8     	   30996	     39159 ns/op	       1 B/op	       0 allocs/op
*/
func BenchmarkPredict(b *testing.B) {
	for _, size := range []int{2, 10, 100, 1000, 10000} {
		b.Run(fmt.Sprintf("10x%dx1", size), func(b *testing.B) {
			nn := NewFeedForward([]int{10, size, 1})
			in := make([]float32, 10)
			out := make([]float32, 1)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				nn.Predict(in, out)
			}
		})
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
	nn := NewFeedForward([]int{3, 2, 1},
		[]float32{-0.27597702, -0.004559007, 0.92628616, -0.5290589, -1.144069, 0.32015398},
		[]float32{1.5643033, 3.2390056},
	)

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
		const delta = 0.01
		out := nn.Predict(tc.input, nil)
		assert.Equal(t, 1, len(out))
		assert.InDelta(t, tc.output, out[0], delta)
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

// axpyRef function, this doesn't use any SIMD as it seems like this version
// is actually faster than blas32 one from gonum
func axpyRef(x, y []float32, alpha float32) {
	_ = y[len(x)-1] // remove bounds checks
	for i, v := range x {
		y[i] += alpha * v
	}
}
