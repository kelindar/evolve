// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package main

import (
	"fmt"
	"math"

	"github.com/kelindar/evolve"
	"github.com/kelindar/evolve/ffnet"
)

// XOR tests
var tests = []struct {
	input  []float32
	output float32
}{
	{input: []float32{1, 0, 0}, output: 0},
	{input: []float32{1, 0, 1}, output: 1},
	{input: []float32{1, 1, 0}, output: 1},
	{input: []float32{1, 1, 1}, output: 0},
}

func main() {
	pop := evolve.New(256, evaluateXOR, func() *ffnet.FeedForward {
		return ffnet.NewFeedForward(3, 2, 1)
	})

	for i := 0; ; i++ { // loop forever
		fittest := pop.Evolve()
		if i%1000 == 0 {
			fmt.Printf("gen %d: best score = %.2f%% %s\n", i,
				evaluateXOR(fittest)/float32(len(tests))*100, fittest.String())
		}
	}
}

func evaluateXOR(g *ffnet.FeedForward) (score float32) {
	for _, tc := range tests {
		out := g.Predict(tc.input, nil)[0]
		score += (1 - float32(math.Abs(float64(out-tc.output))))
	}
	return
}
