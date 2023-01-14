// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package neural

import (
	"encoding/json"
	"sync"

	"github.com/kelindar/evolve"
	"github.com/kelindar/evolve/neural/layer"
	"github.com/kelindar/evolve/neural/math32"
)

// Layer represents a single layer
type Layer interface {
	evolve.Genome
	Update(dst, x *math32.Matrix) *math32.Matrix
	Reset()
}

// Network represents a feed forward neural network
type Network struct {
	mu         sync.Mutex
	shape      []int
	sensorSize int
	outputSize int
	weights    []math32.Matrix
	scratch    [2]math32.Matrix
	layers     []Layer
}

// NewNetwork creates a new NeuralNetwork
func NewNetwork(shape []int, weights ...[]float32) *Network {
	nn := &Network{
		shape:      shape,
		sensorSize: shape[0],
		outputSize: shape[len(shape)-1],
	}

	// Create weight matrices for each prev
	prev := nn.sensorSize
	for _, hidden := range shape[1:] {
		nn.layers = append(nn.layers, layer.NewMGU(prev, hidden))
		prev = hidden
	}

	// Optionally, construct a network from pre-defined values
	for i := range weights {
		mx := nn.weights[i]
		nn.weights[i] = math32.Matrix{
			Rows: mx.Rows,
			Cols: mx.Cols,
			Data: weights[i],
		}
	}
	return nn
}

// Predict performs a forward propagation through the neural network
func (nn *Network) Predict(input, output []float32) []float32 {
	if output == nil {
		output = make([]float32, nn.outputSize)
	}

	// Set the input matrix
	layer := &math32.Matrix{
		Rows: 1, Cols: len(input),
		Data: input,
	}

	nn.mu.Lock()
	defer nn.mu.Unlock()

	for i := range nn.layers {
		layer = nn.layers[i].Update(&nn.scratch[i%2], layer)
	}

	// Copy the output so we can release the lock
	copy(output, layer.Data)
	return output
}

// forward performs M·N matrix multiplication and writes the result to dst after applying ReLU
func (nn *Network) forward(dst, m, n *math32.Matrix) *math32.Matrix {
	dst.Reset(m.Rows, n.Cols)
	math32.Matmul(dst, m, n)
	math32.Lrelu(dst.Data)
	return dst
}

// Crossover performs crossover between two genomes
func (nn *Network) Crossover(g1, g2 evolve.Genome) {
	nn1 := g1.(*Network)
	nn2 := g2.(*Network)

	nn.mu.Lock()
	defer nn.mu.Unlock()

	for i := range nn.layers {
		nn.layers[i].Crossover(nn1.layers[i], nn2.layers[i])
	}
}

// Mutate mutates the genome
func (nn *Network) Mutate() {
	nn.mu.Lock()
	defer nn.mu.Unlock()
	for i := range nn.layers {
		nn.layers[i].Mutate()
	}
}

func (nn *Network) Reset() {
	for i := range nn.layers {
		nn.layers[i].Reset()
	}
}

func (nn *Network) String() string {
	out, _ := json.MarshalIndent(nn.weights, "", "\t")
	return string(out)
}
