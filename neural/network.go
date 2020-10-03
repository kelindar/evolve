// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package neural

import (
	"sort"

	"github.com/kelindar/evolve"
)

// Network represents a neural network.
type Network struct {
	input  int      // Count of input neurons
	output int      // Count of output neurons
	nodes  neurons  // Neurons for the network
	conns  synapses // Synapses, sorted by ID
}

// New creates a function for a random genome string
func New(in, out int) evolve.Genesis {
	origin := newNetwork(in, out)
	return func() evolve.Genome {
		clone := new(Network)
		origin.Clone(clone)
		return clone
	}
}

// newNetwork creates a new neural network.
func newNetwork(inputs, outputs int) *Network {
	nn := &Network{
		input:  inputs,
		output: outputs,
		nodes:  makeNeurons(1+inputs, outputs),
		conns:  make([]synapse, 0, 256),
	}

	// First is always a bias neuron
	nn.nodes[0].value = 1.0
	return nn
}

// Clone clones the neural network by copying all of the neurons and synapses and
// re-assigning the synapse pointers accordingly.
func (n *Network) Clone(dst *Network) {
	defer dst.sort()
	dst.clear()

	// Copy the nodes into the destination
	dst.input = n.input
	dst.output = n.output
	for _, v := range n.nodes {
		dst.nodes = append(dst.nodes, neuron{
			Serial: v.Serial,
			Kind:   v.Kind,
		})
	}

	// Sort the destination nodes so we can find the corresponding ones
	sort.Sort(dst.nodes)
	for _, v := range n.conns {
		dst.conns = append(dst.conns, synapse{
			From:   dst.nodes.Find(v.From.Serial),
			To:     dst.nodes.Find(v.To.Serial),
			Weight: v.Weight,
			Active: v.Active,
		})
	}
}

// Clear clears the network for reuse.
func (n *Network) clear() {
	n.nodes = n.nodes[:0]
	n.conns = n.conns[:0]
}

// sort sorts the connections depending on the neuron and assigns connection slices
// to the appropriate neurons for activation.
func (n *Network) sort() {
	if len(n.conns) == 0 {
		return
	}

	// Sort by neuron ID
	sort.Sort(n.conns)

	// Assign connection slices to neurons. This is basically sub-slicing the main
	// array, so the "Data" pointer of the slice will point to the same underlying
	// array, avoiding extra memory space and allocations.
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

// Crossover applies genetic crossover between two networks. The first parent is
// the fittest of the two.
func (n *Network) Crossover(p1, p2 evolve.Genome) {
	//n1, n2 := p1.(*Network), p2.(*Network)
	//i1, i2 := n1.Last(), n2.Last()

	/*
	 * p1 should have the higher score
	 *  - take all the genes of a
	 *  - if there is a genome in a that is also in b, choose randomly
	 *  - do not take disjoint genes of b
	 *  - take excess genes of a if they exist
	 */

	// Copy of the nodes and synaposes from the fittest into this network
	//n1.Clone(n)

}

// Distance calculates the distance between two neural networks based on their
// genome structure.
func (n *Network) Distance(other *Network) float64 {
	return 0
}
