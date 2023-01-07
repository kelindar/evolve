package layer

import (
	"github.com/kelindar/evolve"
	"github.com/kelindar/evolve/ffnet/math32"
)

type RNN struct {
	Wx math32.Matrix // input weights
	Wh math32.Matrix // hidden state weights
	h  math32.Matrix // hidden state
}

// NewRNN creates a new RNN layer
func NewRNN(inputSize, hiddenSize int) *RNN {
	return &RNN{
		Wx: math32.NewDenseRandom(hiddenSize, inputSize),
		Wh: math32.NewDenseRandom(hiddenSize, hiddenSize),
		h:  math32.NewDense(1, hiddenSize, nil),
	}
}

func (l *RNN) Update(dst, x *math32.Matrix) *math32.Matrix {
	dst.Reset(x.Rows, l.Wh.Cols)

	math32.Matmul(dst, x, &l.Wx)
	math32.Add(dst.Data, l.h.Data)
	math32.Tanh(dst.Data)

	// Remember the hidden state back
	copy(l.h.Data, dst.Data)

	// Feed-forward layer
	dst.Zero()
	math32.Matmul(dst, &l.h, &l.Wh)
	math32.Lrelu(dst.Data)
	return dst
}

// Crossover performs crossover between two genomes
func (l *RNN) Crossover(g1, g2 evolve.Genome) {
	l1 := g1.(*RNN)
	l2 := g2.(*RNN)

	crossoverMatrix(&l.Wx, &l1.Wx, &l2.Wx)
	crossoverMatrix(&l.Wh, &l1.Wh, &l2.Wh)
}

// Mutate mutates the genome
func (l *RNN) Mutate() {
	const rate = 0.05

	mutateVector(l.Wx.Data, rate)
	mutateVector(l.Wh.Data, rate)
}

func (l *RNN) Reset() {
	l.h.Zero()
}
