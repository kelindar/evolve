// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package neural

import (
	"math"
	"sort"

	"github.com/kelindar/evolve"
)

// Network represents a neural network.
type Network struct {
	input  neurons
	hidden neurons
	output neurons
	conns  []synapse
}

// New creates a new neural network.
func New(in, out int) *Network {
	nn := &Network{
		input:  makeNodes(in + 1),
		output: makeNodes(out),
		conns:  make([]synapse, 0, 256),
	}

	// Bias neuron
	nn.input[in].value = 1.0
	return nn
}

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

// sort sorts the connections depending on the neuron and assigns connection slices
// to the appropriate neurons for activation.
func (n *Network) sort() {
	if len(n.conns) == 0 {
		return
	}

	// Sort by neuron ID
	sort.Sort(sortedByNode(n.conns))

	// Assign connection slices to neurons
	prev, lo := n.conns[0].To, 0
	curr, hi := n.conns[0].To, 0
	for i, conn := range n.conns {
		curr, hi = conn.To, i
		if prev != curr {
			prev.Conns = n.conns[lo:hi]
			prev, lo = curr, hi
		}
	}

	// Last neuron
	prev.Conns = n.conns[lo : hi+1]
}

// connect connects two neurons together.
func (n *Network) connect(from, to *neuron, weight float64) {
	defer n.sort() // Keep sorted
	n.conns = append(n.conns, synapse{
		Serial: next(), // Innovation number
		From:   from,   // Left neuron
		To:     to,     // Right neuron
		Weight: weight, // Weight for the connection
		Active: true,   // Default to active
	})
}

// Mutate mutates the network.
func (n *Network) Mutate() {
	defer n.sort() // Keep sorted

}

func (n *Network) Crossover(p1, p2 evolve.Genome) {

}

// Equal checks whether the connection is equal to another connection
/*func (c *conn) Equal(other *conn) bool {
	return c.From == other.From && c.To == other.To
}*/

// https://github.com/Luecx/NEAT/tree/master/vid%209/src

// https://sausheong.github.io/posts/how-to-build-a-simple-artificial-neural-network-with-go/
// https://stats.stackexchange.com/questions/459491/how-do-i-use-matrix-math-in-irregular-neural-networks-generated-from-neuroevolut
