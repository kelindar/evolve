package layer

import (
	"testing"

	"github.com/kelindar/evolve/neural/math32"
	"github.com/stretchr/testify/assert"
)

func TestMGUUpdate(t *testing.T) {
	m := &MGU{
		Wh: math32.NewMatrix(2, 2, []float32{
			1, 0,
			0, 1,
		}),
		Uf: math32.NewMatrix(1, 2, []float32{0.5, 0.5}),
		Bf: math32.NewMatrix(1, 2, []float32{0, 0}),
		Bh: math32.NewMatrix(1, 2, []float32{0, 0}),
		h:  math32.NewMatrix(1, 2, nil),
		hc: math32.NewMatrix(1, 2, nil),
	}

	x := math32.NewMatrix(1, 2, []float32{1, 2})
	var dst math32.Matrix

	// first step
	out1 := m.Update(&dst, &x)
	f := []float32{0.5, 0.5}
	cand := make([]float32, 2)
	math32.Matmul(&math32.Matrix{Rows: 1, Cols: 2, Data: cand}, &x, &m.Wh)
	math32.Tanh(cand)
	expected1 := []float32{
		f[0]*0 + (1-f[0])*cand[0],
		f[1]*0 + (1-f[1])*cand[1],
	}
	assert.InDeltaSlice(t, expected1, out1.Data, 1e-5)

	// second step
	cand2 := make([]float32, 2)
	math32.Matmul(&math32.Matrix{Rows: 1, Cols: 2, Data: cand2}, &x, &m.Wh)
	math32.Tanh(cand2)
	f2 := make([]float32, 2)
	copy(f2, m.h.Data)
	math32.Mul(f2, m.Uf.Data)
	math32.Sigmoid(f2)
	expected2 := []float32{
		f2[0]*out1.Data[0] + (1-f2[0])*cand2[0],
		f2[1]*out1.Data[1] + (1-f2[1])*cand2[1],
	}
	out2 := m.Update(&dst, &x)
	assert.InDeltaSlice(t, expected2, out2.Data, 1e-5)
}
