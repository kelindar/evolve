package ffnet

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

/*
cpu: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz
BenchmarkPredict-8   	 7658848	       154.9 ns/op	       0 B/op	       0 allocs/op
*/
func BenchmarkPredict(b *testing.B) {
	nn := newXOR()
	in := []float64{1, 0, 0}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nn.Predict(in)
	}
}

func TestXOR(t *testing.T) {
	nn := newXOR()
	tests := []struct {
		input  []float64
		output float64
	}{
		{input: []float64{1, 0, 0}, output: 0},
		{input: []float64{1, 0, 1}, output: 1},
		{input: []float64{1, 1, 0}, output: 1},
		{input: []float64{1, 1, 1}, output: 0},
	}
	for _, tc := range tests {
		const delta = 0.05
		out := nn.Predict(tc.input).At(0, 0)
		assert.InDelta(t, tc.output, out, delta)
	}
}

func newXOR() *FeedForward {
	return NewFeedForward(3, 2, 1,
		[]float64{0.32306533513832203, 1.9518742072214708, -1.771002720697405, -0.27034803041215605, -4.104632914191283, 1.9413847794217824},
		[]float64{0.6470581006955358, 1.035718723058992},
	)
}
