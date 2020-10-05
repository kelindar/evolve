// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package neural

import (
	"math"
)

// Predict activates the network
func (n *Network) Predict(input, output []float64) []float64 {
	if output == nil {
		output = make([]float64, n.output)
	}

	// Split in groups
	sensors := n.nodes[1 : 1+n.input]
	outputs := n.nodes[1+n.input : 1+n.input+n.output]
	hidden := n.nodes[1+n.input+n.output:]

	// Set the values for the input neurons
	for i, v := range input {
		sensors[i].value = v
	}

	// Clean the hidden neurons values
	for i := range hidden {
		hidden[i].value = 0
	}

	// Retrieve values and sum up exponentials
	sum := 0.0
	for i, neuron := range outputs {
		v := exp(neuron.Value())
		output[i] = v
		sum += v
	}

	// Normalize
	for i := range output {
		output[i] /= sum
	}
	return output
}

// Value returns the value for the neuron
func (n *neuron) Value() float64 {
	if n.value != 0 || len(n.Conns) == 0 {
		return n.value
	}

	// Sum of the weighted inputs to the neuron
	s := 0.0
	for _, c := range n.Conns {
		if c.Active {
			s += c.Weight * c.From.Value()
		}
	}

	// Keep the value to avoid recalculating
	n.value = swish(s)
	return n.value
}

// Thanks https://codingforspeed.com/using-faster-exponential-approximation/
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

// Swish is the x / (1 + exp(-x)) activation function. Original paper
// https://arxiv.org/abs/1710.05941v1
func swish(x float64) float64 {
	return x / (1.0 + exp(-x))
}

func swish2(x float64) float64 {
	return x / (1.0 + math.Exp(-x))
}

func relu(x float64) float64 {
	// s, e, f mean `sign bit`, `exponent`, and `fractional`, respectively.
	//     seeeffff |     seeeffff
	//     00010100 |     10010100 --+
	// >>        31 |  >>       31   |
	// -------------+-------------   |
	// not 00000000 | not 11111111   |
	// -------------+-------------   |
	//     11111111 |     00000000   |
	// and 00010100 | and 10010100 <-+
	// -------------+-------------
	//     00010100 |     00000000
	// ==         x |          0.0
	v := math.Float64bits(x)
	return math.Float64frombits(v &^ (v >> 63))
}

func relu2(x float64) float64 {
	return math.Max(x, 0)
}
