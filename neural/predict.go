// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package neural

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
	x = 1.0 + x/1024.0
	x *= x
	x *= x
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
