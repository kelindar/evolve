// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package main

import (
	"fmt"
	"math"
	"sync"

	"github.com/kelindar/evolve"
	"github.com/kelindar/evolve/numeric"
)

func main() {
	pop := evolve.New(256, evaluateTanh, numeric.New(1))
	for i := 0; ; i++ { // loop forever
		fittest := pop.Evolve()
		if i%1000 == 0 {
			errors, count := evaluate(fittest)
			fmt.Printf("gen %d: most fit = %s (%.6f%% MSE)\n", i, fittest,
				errors/float64(count))
		}
	}
}

func evaluateTanh(g *numeric.Float32s) float32 {
	errors, count := evaluate(g)

	// invert, so the highest score will be with fewer errors
	score := float32(count) - float32(errors)
	if math.IsNaN(float64(score)) || score <= 0 {
		return 0
	}
	return score
}

// evaluate runs an evaluation
func evaluate(g *numeric.Float32s) (errors float64, count int) {
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for x := -2.0; x <= 2.0; x += 0.05 {
			if int(x*1000) == 0 {
				continue // skip 0.0
			}

			// Compute the tahn32 and math.Tanh
			approx := approxTanh32(g, float32(x))
			expect := math.Tanh(x)

			// Compute the relative error between tahn32 and math.Tanh
			//errors += math.Abs(float64(approx)-expect) / math.Abs(float64(expect))
			errors += math.Pow(float64(approx)-expect, 2)
			count++
		}
	}()
	wg.Wait()
	return
}

// approxTanh32 is an approximation of math.Tanh
func approxTanh32(genome *numeric.Float32s, x float32) float32 {
	c := *genome // constants
	x = x / c[0]
	x = abs32(x+.5) - abs32(x-.5)
	x = (abs32(x) - 2) * x
	return (abs32(x) - 2) * x
}

func abs32(x float32) float32 {
	return math.Float32frombits(math.Float32bits(x) &^ (1 << 31))
}
