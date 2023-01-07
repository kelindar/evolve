package math32

import (
	"errors"
	"math"
	"unsafe"

	"github.com/kelindar/simd"
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

// Matmul multiplies matrix M by N and writes the result into dst
func Matmul(dst, m, n *Matrix) {
	switch {
	case avx2:
		_f32_matmul(
			unsafe.Pointer(&dst.Data[0]), unsafe.Pointer(&m.Data[0]), unsafe.Pointer(&n.Data[0]),
			uint64(m.Rows), uint64(m.Cols), uint64(n.Rows), uint64(n.Cols))
	default:
		_matmul(dst.Data, m.Data, n.Data, m.Rows, m.Cols, n.Rows, n.Cols)
	}
}

// Axpy function (y = ax + y)
func Axpy(x, y []float32, alpha float32) {
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

// ---------------------------------- SIMD ----------------------------------

func Add(dst, src []float32) {
	if len(dst) != len(src) {
		panic("math32: add of different sizes")
	}

	simd.AddFloat32s(dst, dst, src)
}

func Mul(dst, src []float32) {
	if len(dst) != len(src) {
		panic("math32: multiply of different sizes")
	}

	simd.MulFloat32s(dst, dst, src)
}

// ---------------------------------- Activations ----------------------------------

func Sigmoid(x []float32) {
	for i, v := range x {
		x[i] = 1 / (1 + float32(math.Exp(-float64(v))))
	}
}

func Tanh(x []float32) {
	for i, v := range x {
		x[i] = 2/(1+float32(math.Exp(-2*float64(v)))) - 1
	}
}

func Swish(x []float32) {
	for i, v := range x {
		x[i] = v / (1 + float32(math.Exp(-float64(v))))
	}
}

func Lrelu(x []float32) {
	for i, v := range x {
		if v < 0 {
			x[i] = 0.01 * v
		}
	}
}

// ---------------------------------- Matrix ----------------------------------

// Matrix represents a Matrix using the conventional storage scheme.
type Matrix struct {
	Data []float32 `json:"data"`
	Rows int       `json:"rows"`
	Cols int       `json:"cols"`
}

// NewDense creates a new dense matrix
func NewDense(r, c int, data []float32) Matrix {
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

	return Matrix{
		Rows: r,
		Cols: c,
		Data: data,
	}
}

// NewDenseRandom creates a new dense matrix with randomly initialized values
func NewDenseRandom(r, c int) Matrix {
	return NewDense(r, c, randArr(r*c, float64(c)))
}

// Reset resets the matrix to zero and grows it if necessary
func (m *Matrix) Reset(rows, cols int) {
	m.Rows = rows
	m.Cols = cols

	size := rows * cols
	if cap(m.Data) < size {
		m.Data = make([]float32, size)
		return
	}

	// compiles to runtime.memclrNoHeapPointers
	m.Data = m.Data[:size]
	m.Zero()
}

// Zero zeroes the matrix data, but does not change its shape
func (m *Matrix) Zero() {
	Clear(m.Data)
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

// Clear compiles to runtime.memclrNoHeapPointers
func Clear(data []float32) {
	for i := range data {
		data[i] = 0 // cleanup
	}
}
