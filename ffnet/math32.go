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
	r := 1.0 + -x/256.0
	r *= r
	r *= r
	r *= r
	r *= r
	r *= r
	r *= r
	r *= r
	r *= r
	r++
	return x / r
}

// ---------------------------------- Matrix Multiply ----------------------------------

func matmul(dst, m, n []float32, mr, mc, nr, nc int) {
	for i := 0; i < mr; i++ {
		y := dst[i*nc : (i+1)*nc]
		for l, a := range m[i*mc : (i+1)*mc] {
			axpy(a, n[l*nc:(l+1)*nc], y)
		}
	}
}

// axpy function, this doesn't use any SIMD as it seems like this version
// is actually faster than blas32 one from gonum
func axpy(alpha float32, x, y []float32) {
	_ = y[len(x)-1] // remove bounds checks
	for i, v := range x {
		y[i] += alpha * v
	}
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
	/*for i := range m.Data {
		m.Data[i] = 0 // cleanup
	}*/

	for i := 0; i < len(m.Data); i++ {
		m.Data[i] = 0 // cleanup
	}
}

// randomly generate a float64 array
func randArr(size int, v float64) (data []float32) {
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
