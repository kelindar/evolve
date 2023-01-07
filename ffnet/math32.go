package ffnet

import (
	"errors"
	"math"
	"unsafe"

	"github.com/klauspost/cpuid/v2"
	"gonum.org/v1/gonum/stat/distuv"
)

var (
	avx2 = cpuid.CPU.Supports(cpuid.AVX2) && cpuid.CPU.Supports(cpuid.FMA3)
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

// matmul multiplies matrix M by N and writes the result into dst
func matmul(dst, m, n *matrix) {
	switch {
	case avx2:
		_f32_matmul(
			unsafe.Pointer(&dst.Data[0]), unsafe.Pointer(&m.Data[0]), unsafe.Pointer(&n.Data[0]),
			uint64(m.Rows), uint64(m.Cols), uint64(n.Rows), uint64(n.Cols))
	default:
		_matmul(dst.Data, m.Data, n.Data, m.Rows, m.Cols, n.Rows, n.Cols)
	}
}

// axpy function (y = ax + y)
func axpy(x, y []float32, alpha float32) {
	switch {
	case avx2:
		_f32_axpy(unsafe.Pointer(&x[0]), unsafe.Pointer(&y[0]), uint64(len(y)), alpha)
	default:
		_axpy(x, y, alpha)
	}
}

// _matmul function, generic
func _matmul(dst, m, n []float32, mr, mc, nr, nc int) {
	for i := 0; i < mr; i++ {
		y := dst[i*nc : (i+1)*nc]
		for l, a := range m[i*mc : (i+1)*mc] {
			_axpy(n[l*nc:(l+1)*nc], y, a)
		}
	}
}

// _axpy function, generic
func _axpy(x, y []float32, alpha float32) {
	_ = y[len(x)-1] // remove bounds checks
	for i, v := range x {
		y[i] += alpha * v
	}
}

// ---------------------------------- Matrix ----------------------------------

// matrix represents a matrix using the conventional storage scheme.
type matrix struct {
	Data []float32 `json:"data"`
	Rows int       `json:"rows"`
	Cols int       `json:"cols"`
}

// newDense creates a new dense matrix
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

// Reset resets the matrix to zero and grows it if necessary
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
