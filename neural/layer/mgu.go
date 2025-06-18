// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package layer

import (
	"github.com/kelindar/evolve"
	"github.com/kelindar/evolve/neural/math32"
)

// MGU is a minimal gated recurrent unit layer
type MGU struct {
	// Forget gate parameters
	Wf, Uf, Bf math32.Matrix

	// Candidate state parameters
	Wh, Uh, Bh math32.Matrix

	// Internal state and scratch space
	h  math32.Matrix
	hc math32.Matrix
}

// NewMGU creates a new MGU layer, based on https://arxiv.org/abs/1603.09420 and https://arxiv.org/abs/1701.03452
func NewMGU(inputSize, hiddenSize int) *MGU {
	return &MGU{
		Wf: math32.NewMatrixRandom(inputSize, hiddenSize),
		Wh: math32.NewMatrixRandom(inputSize, hiddenSize),
		Uf: math32.NewMatrixRandom(1, hiddenSize),
		Uh: math32.NewMatrixRandom(1, hiddenSize),
		Bf: math32.NewMatrixBias(1, hiddenSize),
		Bh: math32.NewMatrixBias(1, hiddenSize),
		h:  math32.NewMatrix(1, hiddenSize, nil),
		hc: math32.NewMatrix(1, hiddenSize, nil),
	}
}

func (l *MGU) Update(dst, x *math32.Matrix) *math32.Matrix {
	dst.Reset(x.Rows, l.h.Cols)

	// ------------------------------------------------------------------
	// Forget gate: f_t = σ(W_f·x_t + U_f⊙h_{t-1} + b_f)
	// ------------------------------------------------------------------
	math32.Matmul(dst, x, &l.Wf)
	tmp := append([]float32(nil), l.h.Data...)
	math32.Mul(tmp, l.Uf.Data)
	math32.Add(dst.Data, tmp)
	math32.Add(dst.Data, l.Bf.Data)
	math32.Sigmoid(dst.Data)
	f := dst.Data

	// ------------------------------------------------------------------
	// Candidate state: \tilde{h}_t = tanh(W_h·x_t + U_h⊙(f_t⊙h_{t-1}) + b_h)
	// ------------------------------------------------------------------
	hc := &l.hc
	hc.Reset(x.Rows, l.h.Cols)
	math32.Matmul(hc, x, &l.Wh)

	tmp = append(tmp[:0], l.h.Data...)
	math32.Mul(tmp, f)
	math32.Mul(tmp, l.Uh.Data)
	math32.Add(hc.Data, tmp)
	math32.Add(hc.Data, l.Bh.Data)
	math32.Tanh(hc.Data)

	// ------------------------------------------------------------------
	// Final state: h_t = (1-f_t)⊙h_{t-1} + f_t⊙\tilde{h}_t
	// ------------------------------------------------------------------
	for i, ft := range f {
		dst.Data[i] = (1-ft)*l.h.Data[i] + ft*hc.Data[i]
	}

	copy(l.h.Data, dst.Data)
	return dst
}

// Crossover performs crossover between two genomes
func (l *MGU) Crossover(g1, g2 evolve.Genome) {
	l1 := g1.(*MGU)
	l2 := g2.(*MGU)

	crossoverMatrix(&l.Wf, &l1.Wf, &l2.Wf)
	crossoverMatrix(&l.Uf, &l1.Uf, &l2.Uf)
	crossoverMatrix(&l.Bf, &l1.Bf, &l2.Bf)

	crossoverMatrix(&l.Wh, &l1.Wh, &l2.Wh)
	crossoverMatrix(&l.Uh, &l1.Uh, &l2.Uh)
	crossoverMatrix(&l.Bh, &l1.Bh, &l2.Bh)
}

// Mutate mutates the genome
func (l *MGU) Mutate() {
	const rate = 0.05

	mutateWeights(l.Wf.Data, rate)
	mutateWeights(l.Uf.Data, rate)
	mutateBias(l.Bf.Data, rate)

	mutateWeights(l.Wh.Data, rate)
	mutateWeights(l.Uh.Data, rate)
	mutateBias(l.Bh.Data, rate)
}

func (l *MGU) Reset() {
	l.h.Zero()
}
