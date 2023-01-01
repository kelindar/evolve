package ffnet

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

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
