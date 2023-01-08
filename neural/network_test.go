// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package neural

import (
	"fmt"
	"testing"

	"github.com/kelindar/evolve"
	"github.com/stretchr/testify/assert"
)

/*
cpu: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz
BenchmarkPredict/10x2x1-8         	22244012	        53.16 ns/op	       0 B/op	       0 allocs/op
BenchmarkPredict/10x10x1-8        	14422660	        83.48 ns/op	       0 B/op	       0 allocs/op
BenchmarkPredict/10x100x1-8       	 3408278	       350.4 ns/op	       0 B/op	       0 allocs/op
BenchmarkPredict/10x1000x1-8      	  375358	      3116 ns/op	       0 B/op	       0 allocs/op
BenchmarkPredict/10x10000x1-8     	   36583	     32805 ns/op	       1 B/op	       0 allocs/op
*/
func BenchmarkPredict(b *testing.B) {
	for _, size := range []int{2, 10, 100, 1000, 10000} {
		b.Run(fmt.Sprintf("10x%dx1", size), func(b *testing.B) {
			nn := NewNetwork([]int{10, size, 1})
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
BenchmarkEvolve-8   	      16	  67508825 ns/op	       0 B/op	       0 allocs/op
*/
func BenchmarkEvolve(b *testing.B) {
	pop := evolve.New(256, func(*Network) float32 { return 0 }, func() *Network {
		return NewNetwork([]int{3, 128, 128, 1})
	})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pop.Evolve()
	}
}

/*
cpu: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz
BenchmarkDeep-8   	     270	   4218243 ns/op	       3 B/op	       0 allocs/op
*/
func BenchmarkDeep(b *testing.B) {
	const layers = 512
	shape := make([]int, 0, layers)
	shape = append(shape, 10)
	for i := 0; i < layers; i++ {
		shape = append(shape, 128)
	}
	shape = append(shape, 1)

	nn := NewNetwork(shape)
	in := make([]float32, 10)
	out := make([]float32, 1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nn.Predict(in, out)
	}
}

func TestXOR(t *testing.T) {
	nn := NewNetwork([]int{2, 2, 1},
		[]float32{-1.4361037, 0.770241, 0.5583277, -1.5698348},
		[]float32{1.8285279, 1.3325073},
	)

	tests := []struct {
		input  []float32
		output float32
	}{
		{input: []float32{0, 0}, output: 0},
		{input: []float32{0, 1}, output: 1},
		{input: []float32{1, 0}, output: 1},
		{input: []float32{1, 1}, output: 0},
	}
	for _, tc := range tests {
		const delta = 0.01
		out := nn.Predict(tc.input, nil)
		assert.Equal(t, 1, len(out))
		assert.InDelta(t, tc.output, out[0], delta)
	}
}
