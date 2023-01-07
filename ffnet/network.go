// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package ffnet

import (
	"encoding/json"
	"math/rand"
	"sync"

	"github.com/kelindar/evolve"
)

// FeedForward represents a feed forward neural network
type FeedForward struct {
	mu         sync.Mutex
	sensorSize int
	hiddenSize []int
	outputSize int
	weights    []matrix
	scratch    [2]matrix
	memory     []*GRU
}

// NewFeedForward creates a new NeuralNetwork
func NewFeedForward(shape []int, weights ...[]float32) *FeedForward {
	nn := &FeedForward{
		sensorSize: shape[0],
		hiddenSize: shape[1 : len(shape)-1],
		outputSize: shape[len(shape)-1],
	}

	// Create weight matrices for each layer
	layer := nn.sensorSize
	for _, hidden := range nn.hiddenSize {
		nn.weights = append(nn.weights, newDense(layer, hidden, randArr(hidden*layer, float64(hidden))))
		//nn.memory = append(nn.memory, NewGRU(layer, hidden))

		layer = hidden
	}

	nn.weights = append(nn.weights, newDense(layer, nn.outputSize, randArr(nn.outputSize*layer, float64(layer))))
	//nn.memory = append(nn.memory, NewGRU(layer, nn.outputSize))

	// Create a memory layer for each hidden layer
	for i := 1; i < len(shape); i++ {
		nn.memory = append(nn.memory, NewGRU(shape[i], shape[i]))
	}

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
		output = make([]float32, nn.outputSize)
	}

	// Set the input matrix
	layer := &matrix{
		Rows: 1, Cols: len(input),
		Data: input,
	}

	nn.mu.Lock()
	defer nn.mu.Unlock()
	for i := range nn.weights {
		layer = nn.forward(&nn.scratch[i%2], layer, &nn.weights[i])
		layer.Data = nn.memory[i].Update(layer.Data)
	}

	// Copy the output so we can release the lock
	copy(output, layer.Data)
	return output
}

// forward performs M·N matrix multiplication and writes the result to dst after applying ReLU
func (nn *FeedForward) forward(dst, m, n *matrix) *matrix {
	dst.Reset(m.Rows, n.Cols)
	matmul(dst, m, n)

	// Apply activation function (inlined Leaky ReLU)
	for i, x := range dst.Data {
		if x < 0 {
			dst.Data[i] = 0.01 * x
		}
	}
	return dst
}

// Crossover performs crossover between two genomes
func (nn *FeedForward) Crossover(g1, g2 evolve.Genome) {
	nn1 := g1.(*FeedForward)
	nn2 := g2.(*FeedForward)

	nn.mu.Lock()
	defer nn.mu.Unlock()

	for i := range nn.memory {
		nn.memory[i].Crossover(nn1.memory[i], nn2.memory[i])
	}

	for layer := 0; layer < len(nn.weights); layer++ {
		crossoverVector(nn.weights[layer].Data, nn1.weights[layer].Data, nn2.weights[layer].Data)
	}
}

// Mutate mutates the genome
func (nn *FeedForward) Mutate() {
	nn.mu.Lock()
	defer nn.mu.Unlock()
	const rate = 0.01

	for i := range nn.memory {
		nn.memory[i].Mutate()
	}

	for layer := 0; layer < len(nn.weights); layer++ {
		mutateVector(nn.weights[layer].Data)
	}
}

func (nn *FeedForward) String() string {
	out, _ := json.MarshalIndent(nn.weights, "", "\t")
	return string(out)
}

func crossoverMatrix(dst, mx1, mx2 *matrix) {
	crossoverVector(dst.Data, mx1.Data, mx2.Data)
}

func crossoverVector(dst, v1, v2 []float32) {
	clear(dst)
	axpy(v1, dst, .75)
	axpy(v2, dst, .25)
}

func mutateVector(v []float32) {
	const rate = 0.001
	for i, x := range v {
		if rand.Float64() < rate {
			v[i] = x + float32(rand.NormFloat64())
		}
	}
}

// clear compiles to runtime.memclrNoHeapPointers
func clear(data []float32) {
	for i := range data {
		data[i] = 0 // cleanup
	}
}
