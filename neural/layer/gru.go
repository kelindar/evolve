// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package layer

import (
	"github.com/kelindar/evolve"
	"github.com/kelindar/evolve/neural/math32"
)

type GRU struct {
	Wz, Uz, Bz math32.Matrix // Gating unit
	Wh, Uh, Bh math32.Matrix // Hidden unit
	h          math32.Matrix // hidden state
	hc         math32.Matrix // hidden state candidate
}

// NewGRU creates a new GRU layer, based on https://arxiv.org/pdf/1803.04831.pdf
func NewGRU(inputSize, hiddenSize int) *GRU {
	return &GRU{
		Wz: math32.NewDenseRandom(inputSize, hiddenSize),
		Wh: math32.NewDenseRandom(inputSize, hiddenSize),
		Uz: math32.NewDenseRandom(1, hiddenSize),
		Uh: math32.NewDenseRandom(1, hiddenSize),
		Bz: math32.NewDenseRandom(1, hiddenSize),
		Bh: math32.NewDenseRandom(1, hiddenSize),
		h:  math32.NewDense(1, hiddenSize, nil),
		hc: math32.NewDense(1, hiddenSize, nil),
	}
}

func (l *GRU) Update(z, x *math32.Matrix) *math32.Matrix {
	z.Reset(x.Rows, l.Wz.Cols)
	l.hc.Zero()

	// Compute the gating unit
	math32.Matmul(z, x, &l.Wz)
	math32.Mul(l.h.Data, l.Uz.Data)
	math32.Add(z.Data, l.h.Data)
	math32.Add(z.Data, l.Bz.Data)
	math32.Lrelu(z.Data)

	// Compute the hidden unit
	hc := &l.hc
	math32.Matmul(hc, x, &l.Wh)
	math32.Mul(hc.Data, z.Data)
	math32.Mul(hc.Data, l.Uh.Data)
	math32.Add(hc.Data, l.h.Data)
	math32.Add(hc.Data, l.Bh.Data)
	math32.Tanh(hc.Data)

	// Multiply the candidate hidden state by the gating unit
	math32.Mul(hc.Data, z.Data)

	for i := range z.Data {
		z.Data[i] = 1 - z.Data[i]
	}

	math32.Mul(z.Data, l.h.Data)
	math32.Add(z.Data, hc.Data)

	// Remember the hidden state for the next time step
	copy(l.h.Data, z.Data)
	return z
}

// Crossover performs crossover between two genomes
func (l *GRU) Crossover(g1, g2 evolve.Genome) {
	l1 := g1.(*GRU)
	l2 := g2.(*GRU)

	crossoverMatrix(&l.Wh, &l1.Wh, &l2.Wh)
	crossoverMatrix(&l.Uh, &l1.Uh, &l2.Uh)
	crossoverMatrix(&l.Bh, &l1.Bh, &l2.Bh)
	crossoverMatrix(&l.Wz, &l1.Wz, &l2.Wz)
	crossoverMatrix(&l.Uz, &l1.Uz, &l2.Uz)
	crossoverMatrix(&l.Bz, &l1.Bz, &l2.Bz)
}

// Mutate mutates the genome
func (l *GRU) Mutate() {
	const rate = 0.05

	mutateWeights(l.Wh.Data, rate)
	mutateWeights(l.Uh.Data, rate)
	mutateVector(l.Bh.Data, rate)
	mutateWeights(l.Wz.Data, rate)
	mutateWeights(l.Uz.Data, rate)
	mutateVector(l.Bz.Data, rate)
}

func (l *GRU) Reset() {
	l.h.Zero()
}
