// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package layer

import (
	"github.com/kelindar/evolve"
	"github.com/kelindar/evolve/neural/math32"
)

type RNN struct {
	Wx math32.Matrix // input weights
	Wh math32.Matrix // hidden state weights (recurrent weight u)
	Bh math32.Matrix // bias
	h  math32.Matrix // hidden state
}

// NewRNN creates a new RNN layer, based on https://arxiv.org/pdf/1803.04831.pdf
func NewRNN(inputSize, hiddenSize int) *RNN {
	return &RNN{
		Wx: math32.NewMatrixRandom(inputSize, hiddenSize),
		Wh: math32.NewMatrixRandom(1, hiddenSize),
		Bh: math32.NewMatrixBias(1, hiddenSize),
		h:  math32.NewMatrix(1, hiddenSize, nil),
	}
}

func (l *RNN) Update(dst, x *math32.Matrix) *math32.Matrix {
	dst.Reset(x.Rows, l.Wx.Cols)

	// https://github.com/batzner/indrnn/blob/master/ind_rnn_cell.py
	// https://arxiv.org/pdf/1803.04831.pdf
	// ht = σ(Wxt + u·ht−1 + b)
	math32.Matmul(dst, x, &l.Wx)    // (1) = Wxt
	math32.Mul(l.h.Data, l.Wh.Data) // (2) = u·ht−1
	math32.Add(dst.Data, l.h.Data)  // (3) = (1) + (2)
	math32.Add(dst.Data, l.Bh.Data) // (4) = (3) + bias
	math32.Lrelu(dst.Data)          // (5) = σ(4)

	// Remember the hidden state for the next time step
	copy(l.h.Data, dst.Data)
	return dst
}

// Crossover performs crossover between two genomes
func (l *RNN) Crossover(g1, g2 evolve.Genome) {
	l1 := g1.(*RNN)
	l2 := g2.(*RNN)

	crossoverMatrix(&l.Wx, &l1.Wx, &l2.Wx)
	crossoverMatrix(&l.Wh, &l1.Wh, &l2.Wh)
	crossoverMatrix(&l.Bh, &l1.Bh, &l2.Bh)
}

// Mutate mutates the genome
func (l *RNN) Mutate() {
	const rate = 0.05

	mutateWeights(l.Wx.Data, rate)
	mutateWeights(l.Wh.Data, rate)
	mutateVector(l.Bh.Data, rate)
}

func (l *RNN) Reset() {
	l.h.Zero()
}
