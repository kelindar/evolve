package layer

import (
	"testing"

	"github.com/kelindar/evolve/neural/math32"
	"github.com/stretchr/testify/assert"
)

func TestRNNUpdate(t *testing.T) {
	r := &RNN{
		Wx: math32.NewMatrix(2, 2, []float32{
			1, 0,
			0, 1,
		}),
		Wh: math32.NewMatrix(1, 2, []float32{0.5, 0.25}),
		Bh: math32.NewMatrix(1, 2, []float32{0, 0}),
		h:  math32.NewMatrix(1, 2, nil),
	}

	x := math32.NewMatrix(1, 2, []float32{1, 2})
	var dst math32.Matrix

	// first step
	// expected first output
	exp1 := math32.NewMatrix(1, 2, nil)
	math32.Matmul(&exp1, &x, &r.Wx)
	math32.Add(exp1.Data, r.Bh.Data)
	math32.Lrelu(exp1.Data)

	out1 := r.Update(&dst, &x)
	assert.InDeltaSlice(t, exp1.Data, out1.Data, 1e-5)

	// second step with same input
	prev := append([]float32(nil), r.h.Data...)
	exp2 := math32.NewMatrix(1, 2, nil)
	math32.Matmul(&exp2, &x, &r.Wx)
	for i := range prev {
		prev[i] *= r.Wh.Data[i]
	}
	math32.Add(exp2.Data, prev)
	math32.Add(exp2.Data, r.Bh.Data)
	math32.Lrelu(exp2.Data)

	out2 := r.Update(&dst, &x)
	assert.InDeltaSlice(t, exp2.Data, out2.Data, 1e-5)
}
