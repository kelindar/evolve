// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package ffnet

import (
	"fmt"
	"math/rand"
	"sync"

	"github.com/kelindar/evolve"
)

// FeedForward represents a feed forward neural network
type FeedForward struct {
	mu              sync.Mutex
	inputLayerSize  int
	hiddenLayerSize int
	outputLayerSize int
	weights         [2]matrix
	scratch         [2]matrix
}

// NewFeedForward creates a new NeuralNetwork
func NewFeedForward(inputLayerSize, hiddenLayerSize, outputLayerSize int, weights ...[]float32) *FeedForward {
	nn := &FeedForward{
		inputLayerSize:  inputLayerSize,
		hiddenLayerSize: hiddenLayerSize,
		outputLayerSize: outputLayerSize,
	}

	nn.weights[0] = newDense(hiddenLayerSize, inputLayerSize, randomArray(inputLayerSize*hiddenLayerSize, float64(hiddenLayerSize)))
	nn.weights[1] = newDense(outputLayerSize, hiddenLayerSize, randomArray(hiddenLayerSize*outputLayerSize, float64(hiddenLayerSize)))

	// Optionally, construct a network from pre-defined values
	for i := range weights {
		mx := nn.weights[i]
		nn.weights[i] = matrix{
			Rows: mx.Rows,
			Cols: mx.Cols,
			Data: weights[i],
		}
	}
	return nn
}

// Predict performs a forward propagation through the neural network
func (nn *FeedForward) Predict(input, output []float32) []float32 {
	if output == nil {
		output = make([]float32, nn.outputLayerSize)
	}

	// Set the input matrix
	l0 := matrix{
		Rows: len(input), Cols: 1,
		Data: input,
	}

	nn.mu.Lock()
	defer nn.mu.Unlock()
	l1 := nn.forward(&nn.scratch[0], &nn.weights[0], &l0)
	l2 := nn.forward(&nn.scratch[1], &nn.weights[1], l1)

	// Copy the output so we can release the lock
	copy(output, l2.Data)
	return output
}

func (nn *FeedForward) forward(dst, m, n *matrix) *matrix {
	dst.Reset(m.Rows, n.Cols)

	// Perform non-transposed matrix mltiply
	for i := 0; i < m.Rows; i++ {
		y := dst.Data[i : i+n.Cols]
		for l, a := range m.Data[i : i+m.Cols] {
			axpy(a, n.Data[l:l+n.Cols], y)
		}
	}

	// Apply activation function
	for i := 0; i < len(dst.Data); i++ {
		dst.Data[i] = swish(dst.Data[i])
	}
	return dst
}

// axpy function, this doesn't use any SIMD as it seems like this version
// is actually faster than blas32 one from gonum
func axpy(alpha float32, x, y []float32) {
	_ = y[len(x)-1] // remove bounds checks
	for i, v := range x {
		y[i] += alpha * v
	}
}

// Crossover performs crossover between two genomes
func (nn *FeedForward) Crossover(g1, g2 evolve.Genome) {
	nn1 := g1.(*FeedForward)
	nn2 := g2.(*FeedForward)

	nn.mu.Lock()
	defer nn.mu.Unlock()
	for layer := 0; layer < len(nn.weights); layer++ {
		dst := nn.weights[layer].Data
		mx1 := nn1.weights[layer].Data
		mx2 := nn2.weights[layer].Data
		for i := 0; i < len(dst); i++ {
			dst[i] = crossover(mx1[i], mx2[i])
		}
	}
}

// Mutate mutates the genome
func (nn *FeedForward) Mutate() {
	nn.mu.Lock()
	defer nn.mu.Unlock()
	const rate = 0.02

	for layer := 0; layer < len(nn.weights); layer++ {
		dst := nn.weights[layer].Data
		for i := 0; i < len(dst); i++ {
			if rand.Float64() < rate {
				dst[i] = dst[i] + float32(rand.NormFloat64())
			}
		}
	}
}

func (nn *FeedForward) String() string {
	return fmt.Sprintf("{h=%+v, o=%+v}",
		nn.weights[0].Data,
		nn.weights[1].Data,
	)
}

// crossover calculates a crossover between 2 numbers
func crossover(v1, v2 float32) float32 {
	const delta = 0.10
	switch {
	case isNan(v1):
		return v2
	case isNan(v2) || v1 == v2:
		return v1
	default: // e.g. [5, 10], move by x% towards 10
		return v1 + ((v2 - v1) * delta)
	}
}
