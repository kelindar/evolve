package layer

import (
	"github.com/kelindar/evolve"
	"github.com/kelindar/evolve/ffnet/math32"
)

type RNN struct {
	inputSize  int
	hiddenSize int
	//Wxh        matrix
	Whh math32.Matrix
	//Bh      matrix
	h       math32.Matrix
	scratch math32.Matrix
}

// NewRNN creates a new RNN layer
func NewRNN(inputSize, hiddenSize int) *RNN {
	return &RNN{
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		//Wxh:        newDense(hiddenSize, inputSize, randArr(hiddenSize*inputSize, float64(inputSize))),
		Whh: math32.NewDenseRandom(hiddenSize, hiddenSize),
		//Bh:      newDense(hiddenSize, hiddenSize, randArr(hiddenSize*hiddenSize, float64(hiddenSize))),
		h:       math32.NewDense(1, hiddenSize, nil),
		scratch: math32.NewDense(1, hiddenSize, nil),
	}
}

// Update updates the RNN layer
/*func (rnn *RNN) Update(x []float32) []float32 {
	input := newDense(1, len(x), x)

	hh := &rnn.scratch
	hh.Reset(1, rnn.hiddenSize)

	matmul(hh, &rnn.h, &rnn.Whh)
	matmul(hh, &input, &rnn.Wxh)
	add(hh.Data, rnn.Bh.Data)
	tanh(hh.Data)

	rnn.h = *hh
	return hh.Data
}*/

func (rnn *RNN) Update(dst, m *math32.Matrix) *math32.Matrix {
	dst.Reset(m.Rows, rnn.Whh.Cols)

	math32.Matmul(dst, m, &rnn.Whh)
	math32.Add(dst.Data, rnn.h.Data)
	math32.Lrelu(dst.Data)

	// Copy scratch back
	copy(rnn.h.Data, dst.Data)
	return dst
}

// Crossover performs crossover between two genomes
func (rnn *RNN) Crossover(g1, g2 evolve.Genome) {
	rnn1 := g1.(*RNN)
	rnn2 := g2.(*RNN)

	//crossoverMatrix(&rnn.Wxh, &rnn1.Wxh, &rnn2.Wxh)
	crossoverMatrix(&rnn.Whh, &rnn1.Whh, &rnn2.Whh)
	//crossoverMatrix(&rnn.Bh, &rnn1.Bh, &rnn2.Bh)
}

// Mutate mutates the genome
func (rnn *RNN) Mutate() {
	const rate = 0.05
	//mutateVector(rnn.Wxh.Data, rate)
	mutateVector(rnn.Whh.Data, rate)
	//mutateVector(rnn.Bh.Data, rate)
}

func (rnn *RNN) Reset() {
	rnn.h.Reset(1, rnn.hiddenSize)
}
