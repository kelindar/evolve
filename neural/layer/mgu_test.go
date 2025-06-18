package layer

import (
	"testing"

	"github.com/kelindar/evolve/neural/math32"
	"github.com/stretchr/testify/assert"
)

func TestMGUUpdate(t *testing.T) {
	m := &MGU{
		Wf: math32.NewMatrix(2, 2, []float32{
			1, 0,
			0, 1,
		}),
		Wh: math32.NewMatrix(2, 2, []float32{
			1, 0,
			0, 1,
		}),
		Uf: math32.NewMatrix(1, 2, []float32{0.5, 0.5}),
		Uh: math32.NewMatrix(1, 2, []float32{0.1, 0.1}),
		Bf: math32.NewMatrix(1, 2, []float32{0, 0}),
		Bh: math32.NewMatrix(1, 2, []float32{0, 0}),
		h:  math32.NewMatrix(1, 2, nil),
		hc: math32.NewMatrix(1, 2, nil),
	}

	x := math32.NewMatrix(1, 2, []float32{1, 2})
	var dst math32.Matrix

	// first step
	out1 := m.Update(&dst, &x)
	expected1 := manualMGUStep(&m.Wf, &m.Uf, &m.Bf, &m.Wh, &m.Uh, &m.Bh, x.Data, []float32{0, 0})
	assert.InDeltaSlice(t, expected1, out1.Data, 1e-5)

	// second step
	prev := append([]float32(nil), out1.Data...)
	expected2 := manualMGUStep(&m.Wf, &m.Uf, &m.Bf, &m.Wh, &m.Uh, &m.Bh, x.Data, prev)
	out2 := m.Update(&dst, &x)
	assert.InDeltaSlice(t, expected2, out2.Data, 1e-5)
}

func manualMGUStep(Wf, Uf, Bf, Wh, Uh, Bh *math32.Matrix, x, hPrev []float32) []float32 {
	mx := math32.Matrix{Rows: 1, Cols: len(x), Data: append([]float32(nil), x...)}

	f := math32.NewMatrix(1, len(hPrev), nil)
	math32.Matmul(&f, &mx, Wf)
	tmp := append([]float32(nil), hPrev...)
	math32.Mul(tmp, Uf.Data)
	math32.Add(f.Data, tmp)
	math32.Add(f.Data, Bf.Data)
	math32.Sigmoid(f.Data)

	cand := math32.NewMatrix(1, len(hPrev), nil)
	math32.Matmul(&cand, &mx, Wh)
	tmp = append([]float32(nil), hPrev...)
	math32.Mul(tmp, f.Data)
	math32.Mul(tmp, Uh.Data)
	math32.Add(cand.Data, tmp)
	math32.Add(cand.Data, Bh.Data)
	math32.Tanh(cand.Data)

	out := make([]float32, len(hPrev))
	for i := range out {
		out[i] = (1-f.Data[i])*hPrev[i] + f.Data[i]*cand.Data[i]
	}
	return out
}
