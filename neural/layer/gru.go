// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package layer

/*
import (
	"github.com/kelindar/evolve"
	"github.com/kelindar/evolve/neural/math32"
)

// GRU is an implementation of a gated recurrent unit. This implementation follows the approach
// described in the "Simplified Minimal Gated Unit Variations for Recurrent Neural Networks"
// paper by Joel Heck and Fathi M. Salem. The paper proposes a simplified version of GRUs that
// uses minimal gated units, which are a combination of a sigmoid unit and a tanh unit.
type GRU struct {
	Wxr, Wxh,
	Whr, Whh math32.Matrix
	Br, Bz     []float32
	h          math32.Matrix
	scratch    [2]math32.Matrix
	hiddenSize int
}

// NewGRU creates a new GRU layer
func NewGRU(inputSize, hiddenSize int) *GRU {
	layer := &GRU{
		Wxr:        math32.NewDenseRandom(hiddenSize, inputSize),
		Wxh:        math32.NewDenseRandom(hiddenSize, inputSize),
		Whr:        math32.NewDenseRandom(hiddenSize, hiddenSize),
		Whh:        math32.NewDenseRandom(hiddenSize, hiddenSize),
		Br:         randArr(hiddenSize, float64(hiddenSize)),
		Bz:         randArr(hiddenSize, float64(hiddenSize)),
		h:          newDense(1, hiddenSize, nil),
		hiddenSize: hiddenSize,
	}

	for i := range layer.scratch {
		layer.scratch[i] = newDense(1, hiddenSize, nil)
	}
	return layer
}

// Update updates the GRU layer
func (gru *GRU) Update(x []float32) []float32 {
	input := newDense(1, len(x), x)

	hr := gru.scratch[0]
	hh := gru.scratch[1]
	hr.Reset(1, gru.hiddenSize)
	hh.Reset(1, gru.hiddenSize)

	matmul(&hr, &gru.h, &gru.Whr)
	matmul(&hr, &input, &gru.Wxr)
	add(hr.Data, gru.Br)
	sigmoid(hr.Data)

	matmul(&hh, &gru.h, &gru.Whh)
	matmul(&hh, &input, &gru.Wxh)
	add(hh.Data, hh.Data)
	tanh(hh.Data)

	for i, v := range hr.Data {
		hh.Data[i] = hh.Data[i]*v + gru.h.Data[i]*(1-v)
	}
	gru.h = hh
	return hh.Data
}

// Crossover performs crossover between two genomes
func (gru *GRU) Crossover(g1, g2 evolve.Genome) {
	gru1 := g1.(*GRU)
	gru2 := g2.(*GRU)

	crossoverMatrix(&gru.Wxr, &gru1.Wxr, &gru2.Wxr)
	crossoverMatrix(&gru.Wxh, &gru1.Wxh, &gru2.Wxh)
	crossoverMatrix(&gru.Whr, &gru1.Whr, &gru2.Whr)
	crossoverMatrix(&gru.Whh, &gru1.Whh, &gru2.Whh)
	crossoverVector(gru.Br, gru1.Br, gru2.Br)
	crossoverVector(gru.Bz, gru1.Bz, gru2.Bz)
}

// Mutate mutates the genome
func (gru *GRU) Mutate() {
	const rate = 0.02
	mutateVector(gru.Wxr.Data, rate)
	mutateVector(gru.Wxh.Data, rate)
	mutateVector(gru.Whr.Data, rate)
	mutateVector(gru.Whh.Data, rate)
	mutateVector(gru.Br, rate)
	mutateVector(gru.Bz, rate)
}

func (gru *GRU) Reset() {
	gru.h.Reset(1, gru.hiddenSize)
}
*/
