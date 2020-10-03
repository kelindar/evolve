// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package neural

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func BenchmarkPredict(b *testing.B) {
	b.Run("2x2", func(b *testing.B) {
		nn := make2x2()
		in := []float64{1, 0}
		out := []float64{0, 0}

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
	assert.Equal(t, []float64{0.5216145455966438, 0.4783854544033563}, r)
}
