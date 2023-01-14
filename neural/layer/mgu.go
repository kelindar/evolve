// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package layer

import (
	"github.com/kelindar/evolve"
	"github.com/kelindar/evolve/neural/math32"
)

// MGU is a minimal gated recurrent unit layer
type MGU struct {
	Wf, Uf, Bf math32.Matrix // Forget gate
	Wh, Uh, Bh math32.Matrix // Hidden gate
	h          math32.Matrix // hidden state
	hc         math32.Matrix // hidden state candidate
}

// NewMGU creates a new MGU layer, based on https://arxiv.org/abs/1603.09420 and https://arxiv.org/abs/1701.03452
func NewMGU(inputSize, hiddenSize int) *MGU {
	return &MGU{
		Wf: math32.NewDenseRandom(inputSize, hiddenSize),
		Wh: math32.NewDenseRandom(inputSize, hiddenSize),
		Uf: math32.NewDenseRandom(1, hiddenSize),
		Uh: math32.NewDenseRandom(1, hiddenSize),
		Bf: math32.NewDenseRandom(1, hiddenSize),
		Bh: math32.NewDenseRandom(1, hiddenSize),
		h:  math32.NewDense(1, hiddenSize, nil),
		hc: math32.NewDense(1, hiddenSize, nil),
	}
}

func (l *MGU) Update(z, x *math32.Matrix) *math32.Matrix {
	z.Reset(x.Rows, l.Wf.Cols)
	l.hc.Zero()

	// Compute the forget gate
	math32.Matmul(z, x, &l.Wf)
	math32.Mul(z.Data, l.Uf.Data)
	math32.Add(z.Data, l.h.Data)
	math32.Add(z.Data, l.Bf.Data)
	math32.Lrelu(z.Data)

	// Compute the hidden unit
	hc := &l.hc
	math32.Matmul(hc, x, &l.Wh)
	math32.Mul(hc.Data, z.Data)
	math32.Mul(hc.Data, l.Uh.Data)
	math32.Add(hc.Data, l.h.Data)
	math32.Add(hc.Data, l.Bh.Data)
	math32.Tanh(hc.Data)

	// Multiply the candidate hidden state by the forget gate
	math32.Mul(hc.Data, z.Data)

	// Invert the forget gate and multiply it by the previous hidden state
	for i := range z.Data {
		z.Data[i] = 1 - z.Data[i]
	}

	math32.Mul(z.Data, l.h.Data)

	// Add the two together to get the new hidden state
	math32.Add(z.Data, hc.Data)

	// Remember the hidden state for the next time step
	copy(l.h.Data, z.Data)
	return z
}

// Crossover performs crossover between two genomes
func (l *MGU) Crossover(g1, g2 evolve.Genome) {
	l1 := g1.(*MGU)
	l2 := g2.(*MGU)

	crossoverMatrix(&l.Wh, &l1.Wh, &l2.Wh)
	crossoverMatrix(&l.Uh, &l1.Uh, &l2.Uh)
	crossoverMatrix(&l.Bh, &l1.Bh, &l2.Bh)
	crossoverMatrix(&l.Wf, &l1.Wf, &l2.Wf)
	crossoverMatrix(&l.Uf, &l1.Uf, &l2.Uf)
	crossoverMatrix(&l.Bf, &l1.Bf, &l2.Bf)
}

// Mutate mutates the genome
func (l *MGU) Mutate() {
	const rate = 0.05

	mutateWeights(l.Wh.Data, rate)
	mutateWeights(l.Uh.Data, rate)
	mutateVector(l.Bh.Data, rate)
	mutateWeights(l.Wf.Data, rate)
	mutateWeights(l.Uf.Data, rate)
	mutateVector(l.Bf.Data, rate)
}

func (l *MGU) Reset() {
	l.h.Zero()
}
