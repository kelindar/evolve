// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package layer

import (
	"github.com/kelindar/evolve"
	"github.com/kelindar/evolve/neural/math32"
)

// MGU is a minimal gated recurrent unit layer
type MGU struct {
	Uf, Bf math32.Matrix // Forget gate
	Wh, Bh math32.Matrix // Hidden gate
	h, hc  math32.Matrix // hidden state and a candidate
}

// NewMGU creates a new MGU layer, based on https://arxiv.org/abs/1603.09420 and https://arxiv.org/abs/1701.03452
func NewMGU(inputSize, hiddenSize int) *MGU {
	return &MGU{
		Wh: math32.NewMatrixRandom(inputSize, hiddenSize),
		Uf: math32.NewMatrixRandom(1, hiddenSize),
		Bf: math32.NewMatrixBias(1, hiddenSize),
		Bh: math32.NewMatrixBias(1, hiddenSize),
		h:  math32.NewMatrix(1, hiddenSize, nil),
		hc: math32.NewMatrix(1, hiddenSize, nil),
	}
}

func (l *MGU) Update(f, x *math32.Matrix) *math32.Matrix {
	f.Reset(x.Rows, l.h.Cols)
	l.hc.Zero()

	// Compute the forget gate, this is using a simplified
	// version of the forget gate, as described in https://arxiv.org/abs/1701.03452
	copy(f.Data, l.Uf.Data)
	math32.Add(f.Data, l.h.Data)
	math32.Add(f.Data, l.Bf.Data)
	math32.Lrelu(f.Data)

	// Compute the candidate hidden state (h estimate)
	hc := &l.hc
	copy(hc.Data, l.h.Data)
	math32.Mul(hc.Data, f.Data)
	math32.Matmul(hc, x, &l.Wh)
	math32.Add(hc.Data, l.Bh.Data)
	math32.Tanh(hc.Data)

	// Multiply the candidate hidden state by the forget gate
	math32.Mul(hc.Data, f.Data)

	// Invert the forget gate and multiply it by the previous hidden state
	for i := range f.Data {
		f.Data[i] = 1 - f.Data[i]
	}

	math32.Mul(f.Data, l.h.Data)

	// Add the two together to get the new hidden state
	math32.Add(f.Data, hc.Data)

	// Remember the hidden state for the next time step
	copy(l.h.Data, f.Data)
	return f
}

// Crossover performs crossover between two genomes
func (l *MGU) Crossover(g1, g2 evolve.Genome) {
	l1 := g1.(*MGU)
	l2 := g2.(*MGU)

	crossoverMatrix(&l.Wh, &l1.Wh, &l2.Wh)
	crossoverMatrix(&l.Bh, &l1.Bh, &l2.Bh)
	crossoverMatrix(&l.Uf, &l1.Uf, &l2.Uf)
	crossoverMatrix(&l.Bf, &l1.Bf, &l2.Bf)
}

// Mutate mutates the genome
func (l *MGU) Mutate() {
	const rate = 0.05

	mutateWeights(l.Wh.Data, rate)
	mutateWeights(l.Uf.Data, rate)

	mutateBias(l.Bh.Data, rate)
	mutateBias(l.Bf.Data, rate)
}

func (l *MGU) Reset() {
	l.h.Zero()
}
