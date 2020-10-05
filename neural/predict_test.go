// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package neural

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// BenchmarkPredict/10-8         	 2500016	       474 ns/op	       0 B/op	       0 allocs/op
// BenchmarkPredict/100-8        	  272751	      4447 ns/op	       0 B/op	       0 allocs/op
func BenchmarkPredict(b *testing.B) {
	in := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	out := []float64{0}

	b.Run("10", func(b *testing.B) {
		nn := makeNN(10)
		b.ResetTimer()
		b.ReportAllocs()
		for n := 0; n < b.N; n++ {
			nn.Predict(in, out)
		}
	})

	b.Run("100", func(b *testing.B) {
		nn := makeNN(100)
		b.ResetTimer()
		b.ReportAllocs()
		for n := 0; n < b.N; n++ {
			nn.Predict(in, out)
		}
	})
}

func TestPredict(t *testing.T) {
	nn := make2x2()
	i0 := &nn.nodes[1]
	i1 := &nn.nodes[2]
	o0 := &nn.nodes[3]
	o1 := &nn.nodes[4]

	// must be connected
	assert.True(t, i0.connected(o0))
	assert.True(t, i1.connected(o0))
	assert.True(t, i0.connected(o0))
	assert.False(t, i1.connected(o1))

	r := nn.Predict([]float64{0.5, 1}, nil)
	assert.True(t, r[0] > 0.5)
}

func BenchmarkFunc(b *testing.B) {
	var out float64

	b.Run("relu", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for n := 0; n < b.N; n++ {
			out = relu(float64(n))
		}
	})

	b.Run("relu-precise", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for n := 0; n < b.N; n++ {
			out = relu2(float64(n))
		}
	})

	b.Run("swish", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for n := 0; n < b.N; n++ {
			out = swish(float64(n))
		}
	})

	b.Run("swish-precise", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for n := 0; n < b.N; n++ {
			out = swish2(float64(n))
		}
	})

	assert.NotZero(b, out)
}

func TestRelu(t *testing.T) {
	var out1, out2 []float64
	for i := float64(0); i <= 1.0; i += 0.01 {
		out1 = append(out1, relu(i))
		out2 = append(out2, relu2(i))
	}

	assert.Equal(t, out1, out2)
}
