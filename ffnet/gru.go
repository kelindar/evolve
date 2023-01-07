package ffnet

import (
	"math"

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
	Br, Bz     []float32
	h          matrix
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
		h:          newDense(1, hiddenSize, nil),
		hiddenSize: hiddenSize,
	}
}

// Update updates the GRU layer
func (gru *GRU) Update(x []float32) []float32 {
	input := newDense(1, len(x), x)
	hr := newDense(1, gru.hiddenSize, nil)
	hz := newDense(1, gru.hiddenSize, nil)
	hh := newDense(1, gru.hiddenSize, nil)

	matmul(&hr, &gru.h, &gru.Whr)
	matmul(&hr, &input, &gru.Wxr)
	add(hr.Data, gru.Br)
	sigmoid(hr.Data)

	matmul(&hz, &gru.h, &gru.Whz)
	matmul(&hz, &input, &gru.Wxz)
	add(hz.Data, gru.Bz)
	sigmoid(hz.Data)

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
	mutateVector(gru.Wxr.Data)
	mutateVector(gru.Wxz.Data)
	mutateVector(gru.Wxh.Data)
	mutateVector(gru.Whr.Data)
	mutateVector(gru.Whz.Data)
	mutateVector(gru.Whh.Data)
	mutateVector(gru.Br)
	mutateVector(gru.Bz)
}

func add(dst, src []float32) {
	simd.AddFloat32s(dst, dst, src)
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
