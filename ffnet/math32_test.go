package ffnet

import (
	"math"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
)

/*
cpu: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz
BenchmarkAXPY/std-8         	168842340	         7.228 ns/op	       0 B/op	       0 allocs/op
BenchmarkAXPY/asm-8         	189254059	         6.233 ns/op	       0 B/op	       0 allocs/op
*/
func BenchmarkAXPY(b *testing.B) {
	x := []float32{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}
	y := []float32{3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}

	b.Run("std", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			axpyRef(x, y, 3)
		}
	})

	b.Run("asm", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_f32_axpy(
				unsafe.Pointer(&x[0]),
				unsafe.Pointer(&y[0]),
				4, 3.0,
			)
		}
	})
}

func TestApproxSwish(t *testing.T) {
	mae := testApproxSwish(-10, 10)
	assert.InDelta(t, 0, mae, 0.10, "unexpected MAE: %.5f", mae)
}

func testApproxSwish(start, end float32) float64 {
	err, n := 0.0, 0
	for x := start; x <= end; x += 0.05 {
		ratio := referenceSwish(x) / swish(x)
		if referenceSwish(x) == 0 {
			continue // skip
		}

		n++
		err += math.Abs(float64(ratio) - 1)
	}

	// Compute mean absolute error
	return err / float64(n)
}

func referenceSwish(x float32) float32 {
	return x / (1.0 + float32(math.Exp(float64(-x))))
}
