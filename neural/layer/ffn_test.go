package layer

import (
	"testing"

	"github.com/kelindar/evolve/neural/math32"
	"github.com/stretchr/testify/assert"
)

func TestFFNUpdate(t *testing.T) {
	l := &FFN{
		inputSize:  2,
		hiddenSize: 3,
		Wx: math32.NewMatrix(2, 3, []float32{
			1, 2, 3,
			4, 5, 6,
		}),
	}

	x := math32.NewMatrix(1, 2, []float32{1, 2})
	var dst math32.Matrix
	out := l.Update(&dst, &x)

	expected := math32.NewMatrix(1, 3, nil)
	math32.Matmul(&expected, &x, &l.Wx)
	math32.Lrelu(expected.Data)

	assert.Equal(t, expected.Data, out.Data)
}
