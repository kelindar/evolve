package ffnet

import (
	"math"
	"math/rand"

	"github.com/kelindar/evolve"
	"github.com/kelindar/simd"
)

// GRU is an implementation of a gated recurrent unit. This implementation follows the approach
// described in the "Simplified Minimal Gated Unit Variations for Recurrent Neural Networks"
// paper by Joel Heck and Fathi M. Salem. The paper proposes a simplified version of GRUs that
// uses minimal gated units, which are a combination of a sigmoid unit and a tanh unit.
type GRU struct {
	Wxr, Wxz, Wxh,
	Whr, Whz, Whh matrix
	Br, Bz, h  []float32
	hiddenSize int
}

// NewGRU creates a new GRU layer
func NewGRU(inputSize, hiddenSize int) *GRU {
	return &GRU{
		Wxr:        newDense(hiddenSize, inputSize, randArr(hiddenSize*inputSize, float64(inputSize))),
		Wxz:        newDense(hiddenSize, inputSize, randArr(hiddenSize*inputSize, float64(inputSize))),
		Wxh:        newDense(hiddenSize, inputSize, randArr(hiddenSize*inputSize, float64(inputSize))),
		Whr:        newDense(hiddenSize, hiddenSize, randArr(hiddenSize*hiddenSize, float64(hiddenSize))),
		Whz:        newDense(hiddenSize, hiddenSize, randArr(hiddenSize*hiddenSize, float64(hiddenSize))),
		Whh:        newDense(hiddenSize, hiddenSize, randArr(hiddenSize*hiddenSize, float64(hiddenSize))),
		Br:         randArr(hiddenSize, float64(hiddenSize)),
		Bz:         randArr(hiddenSize, float64(hiddenSize)),
		h:          make([]float32, hiddenSize),
		hiddenSize: hiddenSize,
	}
}

// Update updates the GRU layer
func (gru *GRU) Update(x []float32) []float32 {
	hr := make([]float32, gru.hiddenSize)
	hz := make([]float32, gru.hiddenSize)

	_matmul(hr, gru.h, gru.Whr.Data, 1, gru.hiddenSize, gru.Whr.Rows, gru.Whr.Cols)
	_matmul(hr, x, gru.Wxr.Data, 1, len(x), gru.Wxr.Rows, gru.Wxr.Cols)
	add(hr, gru.Br)
	sigmoid(hr)

	_matmul(hz, gru.h, gru.Whz.Data, 1, gru.hiddenSize, gru.Whz.Rows, gru.Whz.Cols)
	_matmul(hz, x, gru.Wxz.Data, 1, len(x), gru.Wxz.Rows, gru.Wxz.Cols)
	add(hz, gru.Bz)
	sigmoid(hz)

	hh := make([]float32, gru.hiddenSize)
	_matmul(hh, gru.h, gru.Whh.Data, 1, gru.hiddenSize, gru.Whh.Rows, gru.Whh.Cols)
	_matmul(hh, x, gru.Wxh.Data, 1, len(x), gru.Wxh.Rows, gru.Wxh.Cols)
	add(hh, hh)
	tanh(hh)

	for i, v := range hr {
		hh[i] = hh[i]*v + gru.h[i]*(1-v)
	}
	gru.h = hh
	return hh
}

// Crossover performs crossover between two genomes
func (gru *GRU) Crossover(g1, g2 evolve.Genome) {
	gru1 := g1.(*GRU)
	gru2 := g2.(*GRU)

	crossoverMatrix(&gru.Wxr, &gru1.Wxr, &gru2.Wxr)
	crossoverMatrix(&gru.Wxz, &gru1.Wxz, &gru2.Wxz)
	crossoverMatrix(&gru.Wxh, &gru1.Wxh, &gru2.Wxh)
	crossoverMatrix(&gru.Whr, &gru1.Whr, &gru2.Whr)
	crossoverMatrix(&gru.Whz, &gru1.Whz, &gru2.Whz)
	crossoverMatrix(&gru.Whh, &gru1.Whh, &gru2.Whh)
	crossoverVector(gru.Br, gru1.Br, gru2.Br)
	crossoverVector(gru.Bz, gru1.Bz, gru2.Bz)
}

// Mutate mutates the genome
func (gru *GRU) Mutate() {
	const rate = 0.01

	mutateMatrix(&gru.Wxr, rate)
	mutateMatrix(&gru.Wxz, rate)
	mutateMatrix(&gru.Wxh, rate)
	mutateMatrix(&gru.Whr, rate)
	mutateMatrix(&gru.Whz, rate)
	mutateMatrix(&gru.Whh, rate)
	mutateVector(gru.Br, rate)
	mutateVector(gru.Bz, rate)
}

func mutateMatrix(mx *matrix, rate float64) {
	for i, v := range mx.Data {
		if rand.Float64() < rate && v != 0 {
			mx.Data[i] = v + float32(rand.NormFloat64())
		}
	}
}

func mutateVector(v []float32, rate float64) {
	for i, x := range v {
		if rand.Float64() < rate && x != 0 {
			v[i] = x + float32(rand.NormFloat64())
		}
	}
}

func add(dst, src []float32) {
	simd.AddFloat32s(dst, dst, src)
	/*for i, v := range src {
		dst[i] += v
	}*/
}

func sigmoid(x []float32) {
	for i, v := range x {
		x[i] = 1 / (1 + float32(math.Exp(-float64(v))))
	}
}

func tanh(x []float32) {
	for i, v := range x {
		x[i] = 2/(1+float32(math.Exp(-2*float64(v)))) - 1
	}
}
