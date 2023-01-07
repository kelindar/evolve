// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package ffnet

import (
	"encoding/json"
	"math/rand"
	"sync"

	"github.com/kelindar/evolve"
	"github.com/kelindar/evolve/ffnet/layer"
	"github.com/kelindar/evolve/ffnet/math32"
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
	sensorSize int
	hiddenSize []int
	outputSize int
	weights    []math32.Matrix
	scratch    [2]math32.Matrix
	layers     []Layer
}

// NewFeedForward creates a new NeuralNetwork
func NewFeedForward(shape []int, weights ...[]float32) *Network {
	nn := &Network{
		sensorSize: shape[0],
		hiddenSize: shape[1 : len(shape)-1],
		outputSize: shape[len(shape)-1],
	}

	// Create weight matrices for each prev
	prev := nn.sensorSize
	for _, hidden := range nn.hiddenSize {
		nn.layers = append(nn.layers, layer.NewFFN(prev, hidden))
		prev = hidden
	}

	// Append the output layer
	nn.layers = append(nn.layers, layer.NewFFN(prev, nn.outputSize))

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
		//layer = nn.forward(&nn.scratch[i%2], layer, &nn.weights[i])
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

func mutateVector(v []float32, rate float64) {
	for i, x := range v {
		if rand.Float64() < rate {
			v[i] = x + float32(rand.NormFloat64())
		}
	}
}

func crossoverMatrix(dst, mx1, mx2 *math32.Matrix) {
	crossoverVector(dst.Data, mx1.Data, mx2.Data)
}

func crossoverVector(dst, v1, v2 []float32) {
	math32.Clear(dst)
	math32.Axpy(v1, dst, .75)
	math32.Axpy(v2, dst, .25)
}
