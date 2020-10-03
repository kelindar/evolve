// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package neural

import (
	"math"
)

// Predict activates the network
func (n *Network) Predict(input, output []float64) []float64 {
	if output == nil {
		output = make([]float64, len(n.output))
	}

	// Set the values for the input neurons
	for i, v := range input {
		n.input[i].value = v
	}

	// Clean the hidden neurons values
	for i := range n.hidden {
		n.hidden[i].value = 0
	}

	// Retrieve values and sum up exponentials
	sum := 0.0
	for i, neuron := range n.output {
		v := math.Exp(neuron.Value())
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
	n.value = sigmoid(s)
	return n.value
}

// Sigmod activation function.
func sigmoid(x float64) float64 {
	return 1.0 / (1 + math.Exp(-x))
}
