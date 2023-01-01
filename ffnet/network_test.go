package ffnet

import (
	"fmt"
	"testing"

	"github.com/kelindar/evolve"
	"github.com/stretchr/testify/assert"
)

/*
cpu: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz
BenchmarkPredict/10x2x1-8         	12393556	        95.21 ns/op	       0 B/op	       0 allocs/op
BenchmarkPredict/10x10x1-8        	 4104460	       288.5 ns/op	       0 B/op	       0 allocs/op
BenchmarkPredict/10x100x1-8       	  479984	      2454 ns/op	       0 B/op	       0 allocs/op
BenchmarkPredict/10x1000x1-8      	   49237	     24248 ns/op	       0 B/op	       0 allocs/op
BenchmarkPredict/10x10000x1-8     	    4789	    240894 ns/op	       8 B/op	       0 allocs/op
*/
func BenchmarkPredict(b *testing.B) {
	for _, size := range []int{2, 10, 100, 1000, 10000} {
		b.Run(fmt.Sprintf("10x%dx1", size), func(b *testing.B) {
			nn := NewFeedForward(10, size, 1)
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
		return NewFeedForward(3, 2, 1)
	})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pop.Evolve()
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

func newXOR() *FeedForward {
	return NewFeedForward(3, 2, 1,
		[]float32{-0.8011746, 1.5051053, -1.7070092, 1.8089261, 63.063934, -88.076096},
		[]float32{2.8215864, 0.5676709},
	)
}
