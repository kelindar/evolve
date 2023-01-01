package ffnet

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/stat/distuv"
)

var (
	errZeroLength        = errors.New("mat: zero length in matrix dimension")
	errNegativeDimension = errors.New("mat: negative dimension")
	errShape             = errors.New("mat: dimension mismatch")
)

func isNan(v float32) bool {
	return v != v
}

func swish(x float32) float32 {
	return x / (1.0 + exp(-x))
}

func exp(x float32) float32 {
	x = 1.0 + x/256.0
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	return x
}

// ---------------------------------- Matrix ----------------------------------

// matrix represents a matrix using the conventional storage scheme.
type matrix struct {
	Data []float32
	Rows int
	Cols int
}

func newDense(r, c int, data []float32) matrix {
	if r <= 0 || c <= 0 {
		if r == 0 || c == 0 {
			panic(errZeroLength)
		}
		panic(errNegativeDimension)
	}

	if data != nil && r*c != len(data) {
		panic(errShape)
	}

	if data == nil {
		data = make([]float32, r*c)
	}

	return matrix{
		Rows: r,
		Cols: c,
		Data: data,
	}
}

func (m *matrix) Reset(rows, cols int) {
	m.Rows = rows
	m.Cols = cols

	size := rows * cols
	if cap(m.Data) < size {
		m.Data = make([]float32, size)
		return
	}

	// compiles to runtime.memclrNoHeapPointers
	m.Data = m.Data[:size]
	for i := range m.Data {
		m.Data[i] = 0 // cleanup
	}
}

// randomly generate a float64 array
func randomArray(size int, v float64) (data []float32) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: +1 / math.Sqrt(v),
	}

	data = make([]float32, size)
	for i := 0; i < size; i++ {
		data[i] = float32(dist.Rand())
	}
	return
}
