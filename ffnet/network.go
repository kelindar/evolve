// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package ffnet

import (
	"fmt"
	"math"
	"math/rand"
	"sync"

	"github.com/kelindar/evolve"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// FeedForward represents a feed forward neural network
type FeedForward struct {
	mu              sync.Mutex
	inputLayerSize  int
	hiddenLayerSize int
	outputLayerSize int
	weights         [2]*mat.Dense
	scratch         [3]*mat.Dense
}

// NewFeedForward creates a new NeuralNetwork
func NewFeedForward(inputLayerSize, hiddenLayerSize, outputLayerSize int, weights ...[]float64) *FeedForward {
	nn := &FeedForward{
		inputLayerSize:  inputLayerSize,
		hiddenLayerSize: hiddenLayerSize,
		outputLayerSize: outputLayerSize,
	}

	nn.weights[0] = mat.NewDense(hiddenLayerSize, inputLayerSize, randomArray(inputLayerSize*hiddenLayerSize, float64(hiddenLayerSize)))
	nn.weights[1] = mat.NewDense(outputLayerSize, hiddenLayerSize, randomArray(hiddenLayerSize*outputLayerSize, float64(hiddenLayerSize)))

	nn.scratch[0] = mat.NewDense(inputLayerSize, 1, nil)
	nn.scratch[1] = mat.NewDense(nn.weights[0].RawMatrix().Rows, nn.scratch[0].RawMatrix().Cols, nil)
	nn.scratch[2] = mat.NewDense(nn.weights[1].RawMatrix().Rows, nn.scratch[1].RawMatrix().Cols, nil)

	// Optionally, construct a network from pre-defined values
	for i := range weights {
		matrix := nn.weights[i].RawMatrix()
		nn.weights[i].SetRawMatrix(blas64.General{
			Rows:   matrix.Rows,
			Cols:   matrix.Cols,
			Stride: matrix.Stride,
			Data:   weights[i],
		})
	}
	return nn
}

// Predict performs a forward propagation through the neural network
func (nn *FeedForward) Predict(inputData []float64) mat.Matrix {
	nn.mu.Lock()
	defer nn.mu.Unlock()

	// Set the input matrix
	nn.scratch[0].SetRawMatrix(blas64.General{
		Rows: len(inputData), Cols: 1, Stride: 1,
		Data: inputData,
	})

	//inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenOutputs := nn.forward(nn.scratch[1], nn.weights[0], nn.scratch[0])
	finalOutputs := nn.forward(nn.scratch[2], nn.weights[1], hiddenOutputs)
	return finalOutputs
}

func (nn *FeedForward) forward(dst *mat.Dense, m, n *mat.Dense) *mat.Dense {
	dst.Zero() // cleanup

	// C = alpha * A * B + beta * C -> equivalent to dst.Product(m,n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, m.RawMatrix(), n.RawMatrix(), 0, dst.RawMatrix())

	// Apply activation function
	matrix := dst.RawMatrix()
	for i := 0; i < len(matrix.Data); i++ {
		matrix.Data[i] = swish(matrix.Data[i])
	}
	return dst
}

// Crossover performs crossover between two genomes
func (nn *FeedForward) Crossover(g1, g2 evolve.Genome) {
	nn1 := g1.(*FeedForward)
	nn2 := g2.(*FeedForward)

	nn.mu.Lock()
	defer nn.mu.Unlock()
	for layer := 0; layer < len(nn.weights); layer++ {
		dst := nn.weights[layer].RawMatrix().Data
		mx1 := nn1.weights[layer].RawMatrix().Data
		mx2 := nn2.weights[layer].RawMatrix().Data
		for i := 0; i < len(dst); i++ {
			dst[i] = crossover(mx1[i], mx2[i])
		}
	}
}

// Mutate mutates the genome
func (nn *FeedForward) Mutate() {
	nn.mu.Lock()
	defer nn.mu.Unlock()
	const rate = 0.05

	for layer := 0; layer < len(nn.weights); layer++ {
		dst := nn.weights[layer].RawMatrix().Data
		for i := 0; i < len(dst); i++ {
			if rand.Float64() < rate {
				dst[i] = dst[i] + rand.NormFloat64()
			}
		}
	}
}

func (nn *FeedForward) String() string {
	return fmt.Sprintf("{h=%+v, o=%+v}",
		nn.weights[0].RawMatrix().Data,
		nn.weights[1].RawMatrix().Data,
	)
}

// crossover calculates a crossover between 2 numbers
func crossover(v1, v2 float64) float64 {
	const delta = 0.10
	switch {
	case math.IsNaN(v1):
		return v2
	case math.IsNaN(v2) || v1 == v2:
		return v1
	default: // e.g. [5, 10], move by x% towards 10
		return v1 + ((v2 - v1) * delta)
	}
}

func swish(x float64) float64 {
	return x / (1.0 + exp(-x))
}

func exp(x float64) float64 {
	x = 1.0 + x/256.0
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	x *= x
	return x
}

// randomly generate a float64 array
func randomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	data = make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}
	return
}
