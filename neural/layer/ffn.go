// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package layer

import (
	"math/rand"

	"github.com/kelindar/evolve"
	"github.com/kelindar/evolve/neural/math32"
)

type FFN struct {
	inputSize  int
	hiddenSize int
	Wx         math32.Matrix // input weights
}

// NewFFN creates a new feed-forward network layer
func NewFFN(inputSize, hiddenSize int) *FFN {
	return &FFN{
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		Wx:         math32.NewMatrixRandom(hiddenSize, hiddenSize),
	}
}

func (l *FFN) Update(dst, x *math32.Matrix) *math32.Matrix {
	dst.Reset(x.Rows, l.Wx.Cols)
	math32.Matmul(dst, x, &l.Wx)
	math32.Lrelu(dst.Data)
	return dst
}

// Crossover performs crossover between two genomes
func (l *FFN) Crossover(g1, g2 evolve.Genome) {
	l1 := g1.(*FFN)
	l2 := g2.(*FFN)

	crossoverMatrix(&l.Wx, &l1.Wx, &l2.Wx)
}

// Mutate mutates the genome
func (l *FFN) Mutate() {
	const rate = 0.05

	mutateWeights(l.Wx.Data, rate)
}

func (l *FFN) Reset() {
	// no recurrent state
}

// ---------------------------------- Evolution ----------------------------------

func crossoverMatrix(dst, mx1, mx2 *math32.Matrix) {
	crossoverVector(dst.Data, mx1.Data, mx2.Data)
}

func crossoverVector(dst, v1, v2 []float32) {
	math32.Clear(dst)
	math32.Axpy(dst, v1, .75)
	math32.Axpy(dst, v2, .25)
}

func mutateVector(v []float32, rate float64) {
	for i, x := range v {
		if rand.Float64() < rate {
			v[i] = x + float32(rand.NormFloat64())
		}
	}
}

func mutateBias(v []float32, rate float64) {
	for i := range v {
		if rand.Float64() < rate {
			v[i] *= float32(rand.NormFloat64() / 100)
		}
	}
}

func mutateWeights(v []float32, rate float64) {
	mutateVector(v, rate)

	/*activateChance := rand.Float64()
	for i, x := range v {
		switch {
		case x == 0 && activateChance <= .00001: // 0.001% chance to activate
			v[i] = float32(rand.NormFloat64())
		case x != 0 && activateChance <= .000001: // 0.0001% chance to disable
			v[i] = 0
		case x != 0 && rand.Float64() <= rate:
			v[i] = x + float32(rand.NormFloat64())
		}
	}*/
}
