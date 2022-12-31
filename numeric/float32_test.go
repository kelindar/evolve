// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package numeric_test

import (
	"math"
	"testing"

	"github.com/kelindar/evolve"
	"github.com/kelindar/evolve/numeric"
	"github.com/stretchr/testify/assert"
)

func TestEvolve(t *testing.T) {
	pop := evolve.New(256, evaluateTanh, numeric.New(1))

	// Evolve
	var last *numeric.Float32s
	for i := 0; i < 500; i++ {
		last = pop.Evolve()
	}

	assert.InDelta(t, 3.4, (*last)[0], 0.1)
}

func evaluateTanh(g *numeric.Float32s) float32 {
	errors, count := 0.0, 0
	for x := -2.0; x <= 2.0; x += 0.1 {
		if int(x*1000) == 0 {
			continue // skip 0.0
		}

		// Compute the tahn32 and math.Tanh
		approx := approxTanh32(g, float32(x))
		expect := math.Tanh(x)

		// Compute the relative error between tahn32 and math.Tanh
		errors += math.Abs(float64(approx)-expect) / math.Abs(float64(expect))
		count++
	}

	// invert, so the highest score will be with fewer errors
	score := float32(count) - float32(errors)
	if math.IsNaN(float64(score)) || score <= 0 {
		return 0
	}
	return score
}

// approxTanh32 is an approximation of math.Tanh
func approxTanh32(genome *numeric.Float32s, x float32) float32 {
	c := *genome // constants
	x = x / c[0]
	x = x * 0.5
	x = abs32(x+0.5) - abs32(x-0.5)
	x = (abs32(x) - 2) * x
	return (abs32(x) - 2) * x
}

func abs32(x float32) float32 {
	return math.Float32frombits(math.Float32bits(x) &^ (1 << 31))
}
